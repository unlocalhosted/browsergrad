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
  type CppCuteAotEntryRequestV2,
  type CppCuteAotJobBodyV2,
  type CppCuteAotJobV2,
  type CppCuteAotSourceFileV2,
  type PreparedCppCuteAotJob,
} from "../../../src/cpp_cute_aot_job.js";
import {
  CPP_CUTE_AOT_RECEIPT_SCHEMA,
  deriveCppCuteAotRunnerReceiptId,
  type CppCuteAotReceiptResourcesV2,
  type CppCuteAotRunnerReceiptBodyV2,
  type CppCuteAotRunnerReceiptV2,
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
  type CppCuteFrontendExtractionLimits,
  type PreparedCppCuteFrontendProfile,
} from "../../../src/cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
  CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
} from "../../../src/cpp_cute_frontend_types.js";
import {
  CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
  CPP_CUTE_FIXTURE_SOURCE_REVISION,
} from "./cpp_cute_frontend_fixtures.js";

const wire = (value: number | bigint): WireU64 => parseWireU64(String(value));

export const PINNED_CPP_CUTE_AOT_JOB_ID = "bg.cpp.aot-job.sha256.d6209a75ffa045f3577ccc5e1b9bf27ba6653cbcb27691fade433ca98770bcc2";
export const PINNED_CPP_CUTE_AOT_INVOCATION_ID = "bg.cpp.aot-invocation.sha256.edfb964e38fada31ac42bb9b926e384e4fd3cf30ee40e8c3ba1cfdc6c6079656";
export const PINNED_CPP_CUTE_AOT_RECEIPT_ID = "bg.cpp.aot-receipt.sha256.d24f53360a204497a79fa64bae074eaa3a4644536ce8132545afdcc4097f40ca";
export const PINNED_CPP_CUTE_AOT_RECEIPT_BYTES_SHA256 = "72ba2d402bd05b59e1087ad58250a03f51c3bf09628c330c88beed9bb936f5c6";
export const PINNED_CPP_CUTE_AOT_RECEIPT_BYTE_LENGTH = "4986";

export interface CppCuteAotReceiptFixture {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly executionEnvironment: PreparedCppCuteAotExecutionEnvironment;
  readonly artifact: VerifiedCppCuteFrontendArtifact;
  readonly artifactResource: VerifiedCppCuteFrontendArtifactResource;
  readonly job: PreparedCppCuteAotJob;
  readonly receipt: CppCuteAotRunnerReceiptV2;
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
    .filter((file) => file.role === "main-source" || file.role === "project-header")
    .map(async (file): Promise<CppCuteAotSourceFileV2> => {
      const body = {
        fileId: `bg.cpp.file.sha256.${"0".repeat(64)}`,
        role: file.role as "main-source" | "project-header",
        virtualPath: file.virtualPath,
        contentSha256: file.contentSha256,
        byteLength: file.byteLength,
      } satisfies CppCuteAotSourceFileV2;
      return { ...body, fileId: await deriveCppCuteAotSourceFileId(body) };
    }));
  sourceFiles.sort((left, right) => left.virtualPath.localeCompare(right.virtualPath));
  const main = sourceFiles.find((file) => file.role === "main-source");
  if (main === undefined) throw new Error("fixture artifact lost main source");
  const source = {
    repository: CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
    revision: CPP_CUTE_FIXTURE_SOURCE_REVISION,
  };
  const anchor = {
    virtualPath: main.virtualPath,
    beginByte: wire(0),
    endByte: wire(1),
    tokenSha256: await sha256Hex(new Uint8Array([0])),
  };
  const outcome = payload.outcome;
  const artifactEntryId = outcome.kind === "accepted"
    ? outcome.selectedEntryIds[0]
    : payload.entries[0]?.entryId;
  if (artifactEntryId === undefined) throw new Error("fixture artifact lost entry");
  const requestBody = {
    requestId: `bg.cpp.entry-request.sha256.${"0".repeat(64)}`,
    kind: "layout" as const,
    declarationKind: "variable" as const,
    anchor,
  } satisfies CppCuteAotEntryRequestV2;
  const request: CppCuteAotEntryRequestV2 = {
    ...requestBody,
    requestId: await deriveCppCuteAotEntryRequestId(requestBody),
  };
  const body = {
    schema: "browsergrad.compiler.cpp-cute.aot-job" as const,
    version: { major: 2 as const, minor: 0 as const },
    profileHash: profile.profileHash,
    source,
    mainVirtualPath: main.virtualPath,
    files: sourceFiles,
    entryRequests: [request],
    expectedOutput: {
      schema: "browsergrad.compiler.cpp-cute.frontend-artifact" as const,
      version: { major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR, minor: CPP_CUTE_FRONTEND_ARTIFACT_MINOR },
      sourceSetSha256: artifact.sourceSetSha256,
      headerSetSha256: artifact.headerSetSha256,
      inputClosureSha256: artifact.inputClosureSha256,
    },
  } satisfies CppCuteAotJobBodyV2;
  const jobValue: CppCuteAotJobV2 = { ...body, jobId: await deriveCppCuteAotJobId(body) };
  const job = await prepareCppCuteAotJob(profile, jobValue);
  const receiptBody = await createCppCuteAotReceiptBody(job, executionEnvironment, artifact);
  const receipt: CppCuteAotRunnerReceiptV2 = {
    ...receiptBody,
    receiptId: await deriveCppCuteAotRunnerReceiptId(receiptBody),
  };
  return { profile, executionEnvironment, artifact, artifactResource, job, receipt };
}

export async function createCppCuteAotReceiptBody(
  job: PreparedCppCuteAotJob,
  executionEnvironment: PreparedCppCuteAotExecutionEnvironment,
  artifact: VerifiedCppCuteFrontendArtifact,
): Promise<CppCuteAotRunnerReceiptBodyV2> {
  const jobRecord = unwrapPreparedCppCuteAotJob(job);
  const profile = jobRecord.profile;
  const configured = unwrapPreparedCppCuteFrontendProfile(profile).profile;
  const artifactRecord = unwrapVerifiedCppCuteFrontendArtifact(artifact);
  const payload = artifactRecord.envelope.payload;
  const request = jobRecord.job.entryRequests[0];
  if (request === undefined) throw new Error("fixture job lost entry request");
  const invocationManifestSha256 = await computeCppCuteAotInvocationManifestHash(job);
  const executionPlanSha256 = await computeCppCuteAotExecutionPlanHash(job, executionEnvironment);
  const resources = createResourceAccounting(artifact, configured.extractionLimits);
  return {
    schema: CPP_CUTE_AOT_RECEIPT_SCHEMA,
    version: { major: 2, minor: 0 },
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
          resolvedEntryId: payload.outcome.selectedEntryIds[0]!,
        }
      : {
          kind: "rejected",
          requestId: request.requestId,
          anchorTokenSha256: request.anchor.tokenSha256,
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

function createResourceAccounting(
  artifact: VerifiedCppCuteFrontendArtifact,
  limits: CppCuteFrontendExtractionLimits,
): CppCuteAotReceiptResourcesV2 {
  const payload = unwrapVerifiedCppCuteFrontendArtifact(artifact).envelope.payload;
  const sources = payload.inputs.files.filter((file) => file.role === "main-source" || file.role === "project-header");
  const headers = payload.inputs.files.filter((file) => file.role !== "main-source" && file.role !== "project-header");
  const sum = (files: typeof payload.inputs.files): bigint => files.reduce(
    (total, file) => total + wireIntegerToBigInt(file.byteLength),
    0n,
  );
  return {
    observedInputs: {
      accountingKind: "observed-exact",
      values: {
        openedSourceFiles: wire(sources.length),
        openedSourceBytes: wire(sum(sources)),
        openedHeaderFiles: wire(headers.length),
        openedHeaderBytes: wire(sum(headers)),
      },
    },
    processMeasurements: {
      accountingKind: "observed-exact",
      values: {
        wallTimeMs: wire(25),
        cpuTimeMs: wire(20),
        peakMemoryBytes: wire(65_536),
        peakProcesses: wire(1),
      },
    },
    emittedArtifact: {
      accountingKind: "emitted-artifact-exact",
      values: {
        macroExpansionFacts: wire(payload.macroExpansions.length),
        templateInstantiationFacts: wire(payload.templateInstantiations.length),
        declarations: wire(payload.declarations.length),
        types: wire(payload.types.length),
        constants: wire(payload.constants.length),
        layoutFacts: wire(payload.facts.filter((fact) => fact.kind === "affine-layout").length),
        tensorFacts: wire(payload.facts.filter((fact) => fact.kind === "tensor").length),
        operationFacts: wire(payload.facts.filter((fact) => (
          fact.kind !== "affine-layout" && fact.kind !== "tensor" && fact.kind !== "target-intrinsic"
        )).length),
        targetIntrinsicFacts: wire(payload.facts.filter((fact) => fact.kind === "target-intrinsic").length),
        diagnostics: wire(payload.diagnostics.length),
        outputBytes: artifact.artifactByteLength,
      },
    },
    enforcedCeilings: {
      accountingKind: "enforced-upper-bound",
      values: {
        maxSourceFiles: wire(limits.maxSourceFiles),
        maxSourceBytes: wire(limits.maxSourceBytes),
        maxHeaderFiles: wire(limits.maxHeaderFiles),
        maxHeaderBytes: wire(limits.maxHeaderBytes),
        maxIncludeDepth: wire(limits.maxIncludeDepth),
        maxMacroExpansions: wire(limits.maxMacroExpansions),
        maxPreprocessedTokens: wire(limits.maxPreprocessedTokens),
        maxAstNodes: wire(limits.maxAstNodes),
        maxConstexprSteps: wire(limits.maxConstexprSteps),
        maxTemplateInstantiations: wire(limits.maxTemplateInstantiations),
        maxTemplateDepth: wire(limits.maxTemplateDepth),
        maxDeclarations: wire(limits.maxDeclarations),
        maxTypes: wire(limits.maxTypes),
        maxConstants: wire(limits.maxConstants),
        maxLayouts: wire(limits.maxLayouts),
        maxTensors: wire(limits.maxTensors),
        maxOperations: wire(limits.maxOperations),
        maxTargetIntrinsics: wire(limits.maxTargetIntrinsics),
        maxDiagnostics: wire(limits.maxDiagnostics),
        maxOutputBytes: wire(limits.maxOutputBytes),
        maxWallTimeMs: wire(limits.maxWallTimeMs),
        maxCpuTimeMs: wire(limits.maxCpuTimeMs),
        maxMemoryBytes: wire(limits.maxMemoryBytes),
        maxProcesses: wire(limits.maxProcesses),
      },
    },
  };
}
