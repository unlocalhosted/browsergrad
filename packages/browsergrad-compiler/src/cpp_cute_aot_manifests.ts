import {
  hashCanonicalJson,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapPreparedCppCuteAotJob,
  type PreparedCppCuteAotJob,
} from "./cpp_cute_aot_job.js";
import {
  unwrapVerifiedCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifact,
} from "./cpp_cute_frontend_artifact.js";
import {
  unwrapPreparedCppCuteAotFrontendProfile,
  type CppCuteFrontendExtractionLimits,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";

/** Hashes the dependency closure declared by one prepared profile. */
export async function computeCppCuteAotDependencyManifestHash(
  profile: PreparedCppCuteFrontendProfile,
): Promise<string> {
  const record = unwrapPreparedCppCuteAotFrontendProfile(profile);
  return hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.profile-dependencies.v1",
    dependencies: record.profile.toolchain.dependencies,
  });
}

/** Hashes the complete closed extraction/sandbox limit set. */
export async function computeCppCuteAotLimitsManifestHash(
  limits: CppCuteFrontendExtractionLimits,
): Promise<string> {
  return hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.extraction-limits.v1",
    limits,
  });
}

/**
 * Computes the invocation before execution. Output facts cannot influence this
 * identity: it is derived only from the prepared job and its exact profile.
 */
export async function computeCppCuteAotInvocationManifestHash(
  job: PreparedCppCuteAotJob,
): Promise<string> {
  const jobRecord = unwrapPreparedCppCuteAotJob(job);
  const profileRecord = unwrapPreparedCppCuteAotFrontendProfile(jobRecord.profile);
  return hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.aot-invocation.v1",
    job: jobRecord.job,
    deployment: profileRecord.profile.deployment,
    language: profileRecord.profile.language,
    target: profileRecord.profile.target,
    toolchain: profileRecord.profile.toolchain,
    virtualFileSystem: profileRecord.profile.virtualFileSystem,
    extractionLimits: profileRecord.profile.extractionLimits,
  });
}

/** Hashes every semantic, transport, and raw resource identity of one output. */
export async function computeCppCuteAotOutputManifestHash(
  artifact: VerifiedCppCuteFrontendArtifact,
): Promise<string> {
  unwrapVerifiedCppCuteFrontendArtifact(artifact);
  return hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.provenance-output.v1",
    artifactId: artifact.artifactId,
    artifactHash: artifact.artifactHash,
    transportHash: artifact.transportHash,
    artifactBytesSha256: artifact.artifactBytesSha256,
    artifactByteLength: artifact.artifactByteLength,
  });
}
