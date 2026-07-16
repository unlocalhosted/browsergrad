import { canonicalizeJson } from "@unlocalhosted/browsergrad-semantic-core/schema";
import type {
  PreparedCppCuteFrontendProfileRecord,
  CppCuteFrontendProfileV2,
} from "./cpp_cute_frontend_profile.js";
import type {
  CppCuteFrontendPayloadV2,
  CppCuteInputOwnerV2,
} from "./cpp_cute_frontend_types.js";

export interface CppCuteFrontendProfileBindingMismatch {
  readonly path: string;
  readonly message: string;
}

/**
 * Checks producer-neutral artifact facts against one exact frontend profile.
 * The caller retains ownership of its domain-specific error and authority type.
 */
export function findCppCuteFrontendProfileBindingMismatch(
  payload: CppCuteFrontendPayloadV2,
  profile: CppCuteFrontendProfileV2,
  compilationContractHash?: string,
): CppCuteFrontendProfileBindingMismatch | null {
  if (compilationContractHash !== undefined && payload.compilationContractHash !== compilationContractHash) {
    return mismatch(
      "$.artifact.compilationContractHash",
      "artifact compilation contract differs from prepared profile",
    );
  }
  for (const [index, artifactPass] of payload.semanticPasses.entries()) {
    const profilePass = profile.language.semanticPasses[index];
    if (profilePass === undefined || artifactPass.ordinal !== profilePass.ordinal ||
        artifactPass.passId !== profilePass.passId || artifactPass.domain !== profilePass.domain ||
        artifactPass.role !== profilePass.role || artifactPass.invocationMode !== profilePass.invocationMode ||
        artifactPass.targetTriple !== profilePass.targetTriple ||
        artifactPass.auxiliaryTargetTriple !== profilePass.auxiliaryTargetTriple ||
        artifactPass.deviceArchitecture !== profilePass.deviceArchitecture) {
      return mismatch(
        `$.artifact.semanticPasses[${index}]`,
        "artifact semantic-pass domain or target differs from prepared profile",
      );
    }
  }
  const artifactRoots = payload.inputs.includeRoots;
  const profileRoots = profile.virtualFileSystem.includeRoots;
  if (artifactRoots.length !== profileRoots.length) {
    return mismatch("$.artifact.inputs.includeRoots", "artifact include-root count differs from profile");
  }
  for (const [index, artifactRoot] of artifactRoots.entries()) {
    const profileRoot = profileRoots[index];
    if (profileRoot === undefined || artifactRoot.includeRootId !== profileRoot.includeRootId ||
        artifactRoot.ordinal !== index || artifactRoot.mode !== profileRoot.mode ||
        artifactRoot.virtualPath !== profileRoot.virtualPath ||
        artifactRoot.manifestSha256 !== profileRoot.manifestSha256 ||
        !sameOwner(artifactRoot.owner, profileRoot.owner)) {
      return mismatch(
        `$.artifact.inputs.includeRoots[${index}]`,
        "artifact include-root identity, precedence, content, or ownership differs from profile",
      );
    }
  }

  const sourceRoots = profile.virtualFileSystem.sourceRoots;
  const includeRoots = new Map(profileRoots.map((root) => [root.includeRootId, root]));
  for (const [index, file] of payload.inputs.files.entries()) {
    if (file.owner.kind === "source") {
      if (!sourceRoots.some((root) => virtualPathContains(root, file.virtualPath))) {
        return mismatch(
          `$.artifact.inputs.files[${index}].virtualPath`,
          "source-owned file escapes profile source roots",
        );
      }
      continue;
    }
    const root = file.includeRootId === null ? undefined : includeRoots.get(file.includeRootId);
    if (root === undefined || !sameOwner(file.owner, root.owner) ||
        !virtualPathContains(root.virtualPath, file.virtualPath)) {
      return mismatch(
        `$.artifact.inputs.files[${index}]`,
        "toolchain-owned file differs from its profile include-root authority",
      );
    }
  }

  const forcedEdges = payload.inputs.includeEdges.flatMap((edge, artifactEdgeIndex) => (
    edge.kind === "compiler-forced" ? [{ edge, artifactEdgeIndex }] : []
  ));
  const forcedOptions = profile.language.options.flatMap((option, compilerOptionOrdinal) => (
    option.kind === "forced-include" ? [{ option, compilerOptionOrdinal }] : []
  ));
  if (forcedEdges.length !== forcedOptions.length) {
    return mismatch(
      "$.artifact.inputs.includeEdges",
      "compiler-forced include edges must exactly match profile forced-include options",
    );
  }
  const seenOrdinals = new Set<number>();
  for (const { edge, artifactEdgeIndex } of forcedEdges) {
    const path = `$.artifact.inputs.includeEdges[${artifactEdgeIndex}]`;
    if (seenOrdinals.has(edge.compilerOptionOrdinal)) {
      return mismatch(`${path}.compilerOptionOrdinal`, "compiler-forced option ordinal must be unique");
    }
    seenOrdinals.add(edge.compilerOptionOrdinal);
    const option = profile.language.options[edge.compilerOptionOrdinal];
    const file = payload.inputs.files.find((candidate) => candidate.fileId === edge.fileId);
    if (option?.kind !== "forced-include" || file === undefined ||
        edge.includeRootId !== option.includeRootId || file.includeRootId !== option.includeRootId ||
        file.virtualPath !== option.virtualPath) {
      return mismatch(path, "compiler-forced include differs from its exact profile option and file");
    }
  }
  for (const { compilerOptionOrdinal } of forcedOptions) {
    if (!seenOrdinals.has(compilerOptionOrdinal)) {
      return mismatch(
        "$.artifact.inputs.includeEdges",
        `artifact omits forced-include compiler option ordinal ${compilerOptionOrdinal}`,
      );
    }
  }
  return null;
}

export function findCppCutePreparedFrontendProfileBindingMismatch(
  payload: CppCuteFrontendPayloadV2,
  profileRecord: PreparedCppCuteFrontendProfileRecord,
): CppCuteFrontendProfileBindingMismatch | null {
  return findCppCuteFrontendProfileBindingMismatch(
    payload,
    profileRecord.profile,
    profileRecord.compilationContractHash,
  );
}

function sameOwner(left: CppCuteInputOwnerV2, right: CppCuteInputOwnerV2): boolean {
  return canonicalizeJson(left) === canonicalizeJson(right);
}

function virtualPathContains(root: string, candidate: string): boolean {
  return root === "/" ? candidate.startsWith("/") && candidate !== "/" : candidate.startsWith(`${root}/`);
}

function mismatch(path: string, message: string): CppCuteFrontendProfileBindingMismatch {
  return { path, message };
}
