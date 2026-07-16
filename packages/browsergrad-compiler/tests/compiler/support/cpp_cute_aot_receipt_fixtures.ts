import {
  parseWireU64,
  sha256Hex,
  wireIntegerToBigInt,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  computeCppCuteAotDependencyManifestHash,
  computeCppCuteAotInvocationManifestHash,
  computeCppCuteAotLimitsManifestHash,
  computeCppCuteAotOutputManifestHash,
} from "../../../src/cpp_cute_aot_manifests.js";
import { computeCppCuteAotExecutionPlanHash } from "../../../src/cpp_cute_aot_policy.js";
import type { PreparedCppCuteAotExecutionEnvironment } from "../../../src/cpp_cute_aot_environment.js";
import {
  deriveCppCuteAotEntryRequestId,
  deriveCppCuteAotJobId,
  deriveCppCuteAotSourceFileId,
  prepareCppCuteAotJob,
  unwrapPreparedCppCuteAotJob,
  type CppCuteAotEntryRequestV1,
  type CppCuteAotJobBodyV1,
  type CppCuteAotJobV1,
  type CppCuteAotSourceFileV1,
  type PreparedCppCuteAotJob,
} from "../../../src/cpp_cute_aot_job.js";
import {
  CPP_CUTE_AOT_RECEIPT_SCHEMA,
  deriveCppCuteAotRunnerReceiptId,
  type CppCuteAotReceiptResourcesV1,
  type CppCuteAotRunnerReceiptBodyV1,
  type CppCuteAotRunnerReceiptV1,
} from "../../../src/cpp_cute_aot_receipt.js";
import {
  canonicalCppCuteFrontendArtifactBytes,
  decodeCppCuteFrontendArtifact,
  unwrapVerifiedCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifactResource,
} from "../../../src/cpp_cute_frontend_artifact.js";
import {
  unwrapPreparedCppCuteFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../../../src/cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
  CPP_CUTE_FIXTURE_SOURCE_REVISION,
} from "./cpp_cute_frontend_fixtures.js";

const wire = (value: number | bigint): WireU64 => parseWireU64(String(value));

export const PINNED_CPP_CUTE_AOT_JOB_ID = "bg.cpp.aot-job.sha256.0ae1890711adda08e283ba10d736472e335d845bbe730da05d759eb2cd9f5a49";
export const PINNED_CPP_CUTE_AOT_INVOCATION_ID = "bg.cpp.aot-invocation.sha256.6b9a33bfe556442addc62fbb680a67a23d8d4cc0b7de94b37d37f1348d668a7f";
export const PINNED_CPP_CUTE_AOT_RECEIPT_ID = "bg.cpp.aot-receipt.sha256.4168a9281075454a03bfb3d1b73a5b277d7f9c60c00f477e0b89b2244e9964fe";
export const PINNED_CPP_CUTE_AOT_RECEIPT_BYTES_SHA256 = "29e3e1c54d4f8272b4ef569c144c5720e0993e4b4e80ecc50bc2d84dfc979acc";
export const PINNED_CPP_CUTE_AOT_RECEIPT_BYTE_LENGTH = "4131";

export interface CppCuteAotReceiptFixture {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly executionEnvironment: PreparedCppCuteAotExecutionEnvironment;
  readonly artifact: VerifiedCppCuteFrontendArtifact;
  readonly artifactResource: VerifiedCppCuteFrontendArtifactResource;
  readonly job: PreparedCppCuteAotJob;
  readonly receipt: CppCuteAotRunnerReceiptV1;
}

export async function createCppCuteAotReceiptFixture(
  profile: PreparedCppCuteFrontendProfile,
  executionEnvironment: PreparedCppCuteAotExecutionEnvironment,
  artifact: VerifiedCppCuteFrontendArtifact,
): Promise<CppCuteAotReceiptFixture> {
  const artifactResource = await decodeCppCuteFrontendArtifact(canonicalCppCuteFrontendArtifactBytes(artifact));
  const artifactRecord = unwrapVerifiedCppCuteFrontendArtifact(artifact);
  const payload = artifactRecord.envelope.payload;
  const sourceFiles = await Promise.all(payload.inputs.files
    .filter((file) => file.profileDependency === "none")
    .map(async (file): Promise<CppCuteAotSourceFileV1> => {
      const body = {
        fileId: `bg.cpp.file.sha256.${"0".repeat(64)}`,
        role: file.role as "main-source" | "project-header",
        virtualPath: file.virtualPath,
        contentSha256: file.contentSha256,
        byteLength: file.byteLength,
      } satisfies CppCuteAotSourceFileV1;
      return { ...body, fileId: await deriveCppCuteAotSourceFileId(body) };
    }));
  sourceFiles.sort((left, right) => left.virtualPath.localeCompare(right.virtualPath));
  const main = sourceFiles.find((file) => file.role === "main-source");
  if (main === undefined) throw new Error("fixture artifact lost main source");
  const outcome = payload.outcome;
  const expectedEntryId = outcome.kind === "accepted"
    ? outcome.selectedEntryIds[0]
    : payload.entries[0]?.entryId;
  if (expectedEntryId === undefined) throw new Error("fixture artifact lost expected entry");
  const requestBody = {
    requestId: `bg.cpp.entry-request.sha256.${"0".repeat(64)}`,
    expectedEntryId,
    kind: "layout" as const,
    declarationKind: "variable" as const,
    anchor: {
      virtualPath: main.virtualPath,
      beginByte: wire(0),
      endByte: wire(1),
      tokenSha256: await sha256Hex(new Uint8Array([0])),
    },
  } satisfies CppCuteAotEntryRequestV1;
  const request: CppCuteAotEntryRequestV1 = {
    ...requestBody,
    requestId: await deriveCppCuteAotEntryRequestId(requestBody),
  };
  const body = {
    schema: "browsergrad.compiler.cpp-cute.aot-job" as const,
    version: { major: 1 as const, minor: 0 as const },
    profileHash: profile.profileHash,
    source: {
      repository: CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
      revision: CPP_CUTE_FIXTURE_SOURCE_REVISION,
    },
    mainVirtualPath: main.virtualPath,
    files: sourceFiles,
    entryRequests: [request],
    expectedOutput: {
      schema: "browsergrad.compiler.cpp-cute.frontend-artifact" as const,
      version: { major: 1 as const, minor: 0 as const },
      sourceSetSha256: artifact.sourceSetSha256,
      headerSetSha256: artifact.headerSetSha256,
      inputClosureSha256: artifact.inputClosureSha256,
    },
  } satisfies CppCuteAotJobBodyV1;
  const jobValue: CppCuteAotJobV1 = { ...body, jobId: await deriveCppCuteAotJobId(body) };
  const job = await prepareCppCuteAotJob(profile, jobValue);
  const receiptBody = await createCppCuteAotReceiptBody(job, executionEnvironment, artifact);
  const receipt: CppCuteAotRunnerReceiptV1 = {
    ...receiptBody,
    receiptId: await deriveCppCuteAotRunnerReceiptId(receiptBody),
  };
  return { profile, executionEnvironment, artifact, artifactResource, job, receipt };
}

export async function createCppCuteAotReceiptBody(
  job: PreparedCppCuteAotJob,
  executionEnvironment: PreparedCppCuteAotExecutionEnvironment,
  artifact: VerifiedCppCuteFrontendArtifact,
): Promise<CppCuteAotRunnerReceiptBodyV1> {
  const jobRecord = unwrapPreparedCppCuteAotJob(job);
  const profile = jobRecord.profile;
  const configured = unwrapPreparedCppCuteFrontendProfile(profile).profile;
  const artifactRecord = unwrapVerifiedCppCuteFrontendArtifact(artifact);
  const payload = artifactRecord.envelope.payload;
  const request = jobRecord.job.entryRequests[0];
  if (request === undefined) throw new Error("fixture job lost entry request");
  const invocationManifestSha256 = await computeCppCuteAotInvocationManifestHash(job);
  const executionPlanSha256 = await computeCppCuteAotExecutionPlanHash(job, executionEnvironment);
  const resources = createResourceObservations(artifact);
  return {
    schema: CPP_CUTE_AOT_RECEIPT_SCHEMA,
    version: { major: 1, minor: 1 },
    jobId: job.jobId,
    profileHash: profile.profileHash,
    invocation: {
      invocationId: `bg.cpp.aot-invocation.sha256.${invocationManifestSha256}`,
      invocationManifestSha256,
      executionPlanSha256,
      executionEnvironmentManifestSha256: executionEnvironment.manifestSha256,
      runner: configured.deployment.runner,
      container: configured.deployment.container,
      extractor: configured.deployment.extractor,
      compiler: configured.toolchain.compiler,
      dependencyManifestSha256: await computeCppCuteAotDependencyManifestHash(profile),
      sandbox: {
        contractId: configured.deployment.contractId,
        policySha256: configured.deployment.sandboxPolicySha256,
        limitsSha256: await computeCppCuteAotLimitsManifestHash(configured.extractionLimits),
        network: "none",
        readOnlyRoot: true,
        noNewPrivileges: true,
        linking: "forbidden",
        userProducedNativeExecution: "forbidden",
      },
    },
    openedInputs: {
      files: jobRecord.job.files,
      sourceSetSha256: jobRecord.job.expectedOutput.sourceSetSha256,
      headerSetSha256: jobRecord.job.expectedOutput.headerSetSha256,
      inputClosureSha256: jobRecord.job.expectedOutput.inputClosureSha256,
    },
    selection: payload.outcome.kind === "accepted"
      ? {
          kind: "resolved",
          requestId: request.requestId,
          anchorTokenSha256: request.anchor.tokenSha256,
          expectedEntryId: request.expectedEntryId,
          resolvedEntryId: request.expectedEntryId,
        }
      : {
          kind: "rejected",
          requestId: request.requestId,
          anchorTokenSha256: request.anchor.tokenSha256,
          expectedEntryId: request.expectedEntryId,
          blockingDiagnosticIds: payload.outcome.blockingDiagnosticIds,
        },
    output: {
      artifactId: artifact.artifactId,
      artifactHash: artifact.artifactHash,
      transportHash: artifact.transportHash,
      artifactBytesSha256: artifact.artifactBytesSha256,
      artifactByteLength: artifact.artifactByteLength,
      outputManifestSha256: await computeCppCuteAotOutputManifestHash(artifact),
    },
    resources,
    outcome: "succeeded",
    exitCode: 0,
  };
}

function createResourceObservations(artifact: VerifiedCppCuteFrontendArtifact): CppCuteAotReceiptResourcesV1 {
  const payload = unwrapVerifiedCppCuteFrontendArtifact(artifact).envelope.payload;
  const sources = payload.inputs.files.filter((file) => file.profileDependency === "none");
  const headers = payload.inputs.files.filter((file) => file.profileDependency !== "none");
  const sum = (files: typeof payload.inputs.files): bigint => files.reduce(
    (total, file) => total + wireIntegerToBigInt(file.byteLength),
    0n,
  );
  return {
    sourceFiles: wire(sources.length),
    sourceBytes: wire(sum(sources)),
    headerFiles: wire(headers.length),
    headerBytes: wire(sum(headers)),
    includeDepth: wire(1),
    macroExpansions: wire(payload.macroExpansions.length),
    preprocessedTokens: wire(100),
    astNodes: wire(100),
    constexprSteps: wire(10),
    templateInstantiations: wire(payload.templateInstantiations.length),
    templateDepth: wire(1),
    declarations: wire(payload.declarations.length),
    types: wire(payload.types.length),
    constants: wire(payload.constants.length),
    layouts: wire(payload.facts.filter((fact) => fact.kind === "affine-layout").length),
    tensors: wire(payload.facts.filter((fact) => fact.kind === "tensor").length),
    operations: wire(payload.facts.filter((fact) => (
      fact.kind !== "affine-layout" && fact.kind !== "tensor" && fact.kind !== "target-intrinsic"
    )).length),
    targetIntrinsics: wire(payload.facts.filter((fact) => fact.kind === "target-intrinsic").length),
    diagnostics: wire(payload.diagnostics.length),
    outputBytes: artifact.artifactByteLength,
    wallTimeMs: wire(25),
    cpuTimeMs: wire(20),
    peakMemoryBytes: wire(65_536),
    peakProcesses: wire(1),
  };
}
