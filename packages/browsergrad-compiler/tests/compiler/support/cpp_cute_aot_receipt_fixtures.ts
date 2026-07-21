import {
  encodeWireU64,
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
import {
  CPP_CUTE_AOT_RUN_METADATA_SCHEMA,
  CPP_CUTE_GIT_SOURCE_REFERENCE_SCHEMA,
  deriveCppCuteAotRunMetadataHash,
  deriveCppCuteGitSourceReferenceStatementSha256,
  prepareCppCuteAotRunMetadata,
  unwrapPreparedCppCuteAotRunMetadata,
  type CppCuteAotRunMetadataBodyV1,
  type CppCuteAotRunMetadataV1,
  type CppCuteGitSourceReferenceStatementV1,
  type PreparedCppCuteAotRunMetadata,
} from "../../../src/cpp_cute_aot_run_metadata.js";
import { computeCppCuteAotExecutionPlanHash } from "../../../src/cpp_cute_aot_policy.js";
import type { PreparedCppCuteAotExecutionEnvironment } from "../../../src/cpp_cute_aot_environment.js";
import {
  CPP_CUTE_AOT_RECEIPT_SCHEMA,
  deriveCppCuteAotRunnerReceiptId,
  type CppCuteAotReceiptResourcesV3,
  type CppCuteAotRunnerReceiptBodyV3,
  type CppCuteAotRunnerReceiptV3,
} from "../../../src/cpp_cute_aot_receipt.js";
import {
  canonicalCppCuteFrontendArtifactBytes,
  decodeCppCuteFrontendArtifact,
  deriveCppCuteFrontendArtifactId,
  verifyCppCuteFrontendArtifact,
  unwrapVerifiedCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifactResource,
} from "../../../src/cpp_cute_frontend_artifact.js";
import {
  CPP_CUTE_FRONTEND_REQUEST_LOGICAL_GEMM_TILE_MINOR,
  CPP_CUTE_FRONTEND_REQUEST_SCHEMA,
  deriveCppCuteFrontendEntryRequestId,
  deriveCppCuteFrontendRequestHash,
  deriveCppCuteFrontendSourceFileId,
  prepareCppCuteFrontendRequest,
  type CppCuteFrontendEntryRequestV1,
  type CppCuteFrontendRequestBodyV1,
  type CppCuteFrontendRequestLimitsV1,
  type CppCuteFrontendRequestV1,
  type CppCuteFrontendSourceSnapshotInput,
  type PreparedCppCuteFrontendRequest,
} from "../../../src/cpp_cute_frontend_request.js";
import {
  prepareCppCuteFrontendRequestBinding,
  type PreparedCppCuteFrontendRequestBinding,
} from "../../../src/cpp_cute_frontend_request_binding.js";
import {
  unwrapPreparedCppCuteAotFrontendProfile,
  type CppCuteFrontendExtractionLimits,
  type PreparedCppCuteFrontendProfile,
} from "../../../src/cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
  CPP_CUTE_FRONTEND_ARTIFACT_LOGICAL_GEMM_TILE_MINOR,
  CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
  type CppCuteFrontendArtifactV3,
  type CppCuteFrontendPayloadV3,
} from "../../../src/cpp_cute_frontend_types.js";
import {
  computeCppCuteInputHashes,
  computeCppCuteSemanticPassInputClosureHash,
  computeCppCuteSharedSurfaceHash,
} from "../../../src/cpp_cute_frontend_verify.js";
import {
  CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
  CPP_CUTE_FIXTURE_SOURCE_REVISION,
  createCppCuteArtifactInput,
  rebindCppCuteFixtureSourceEntityIds,
} from "./cpp_cute_frontend_fixtures.js";

const wire = (value: number | bigint): WireU64 => parseWireU64(String(value));

export interface CppCuteAotReceiptFixture {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly executionEnvironment: PreparedCppCuteAotExecutionEnvironment;
  readonly artifact: VerifiedCppCuteFrontendArtifact;
  readonly artifactResource: VerifiedCppCuteFrontendArtifactResource;
  readonly request: PreparedCppCuteFrontendRequest;
  readonly metadata: PreparedCppCuteAotRunMetadata;
  readonly requestBinding: PreparedCppCuteFrontendRequestBinding;
  readonly sourceSnapshots: readonly CppCuteFrontendSourceSnapshotInput[];
  readonly receipt: CppCuteAotRunnerReceiptV3;
}

export async function createCppCuteAotReceiptFixture(
  profile: PreparedCppCuteFrontendProfile,
  executionEnvironment: PreparedCppCuteAotExecutionEnvironment,
  outcome: "accepted" | "rejected" = "accepted",
  mutatePayload?: (payload: CppCuteFrontendPayloadV3) => void | Promise<void>,
): Promise<CppCuteAotReceiptFixture> {
  const sourceBytes = cppCuteAotFixtureSourceBytes();
  const artifactInput = await createRealSourceBackedArtifact(
    profile.compilationContractHash,
    sourceBytes,
    outcome,
    mutatePayload,
  );
  const artifact = await verifyCppCuteFrontendArtifact(artifactInput);
  const artifactResource = await decodeCppCuteFrontendArtifact(canonicalCppCuteFrontendArtifactBytes(artifact));
  const payload = unwrapVerifiedCppCuteFrontendArtifact(artifact).envelope.payload;
  const main = payload.inputs.files.find((file) => file.role === "main-source" && file.owner.kind === "source");
  if (main === undefined) throw new Error("fixture artifact lost main source");
  const sourceSnapshots = Object.freeze([{ virtualPath: main.virtualPath, bytes: sourceBytes }]);
  const request = await createRequest(profile, payload, main, sourceBytes);
  const statement: CppCuteGitSourceReferenceStatementV1 = {
    schema: CPP_CUTE_GIT_SOURCE_REFERENCE_SCHEMA,
    version: { major: 1, minor: 0 },
    repository: CPP_CUTE_FIXTURE_SOURCE_REPOSITORY,
    revision: CPP_CUTE_FIXTURE_SOURCE_REVISION,
  };
  const statementSha256 = await deriveCppCuteGitSourceReferenceStatementSha256(statement);
  const preparedRequest = await prepareCppCuteFrontendRequest(profile, request, sourceSnapshots, {
    detached: { declaredSourceReference: { statementSha256 }, conformance: null },
  });
  const metadataBody: CppCuteAotRunMetadataBodyV1 = {
    schema: CPP_CUTE_AOT_RUN_METADATA_SCHEMA,
    version: { major: 1, minor: 0 },
    profileHash: profile.profileHash,
    requestId: preparedRequest.requestId,
    declaredSourceReference: { statementSha256, statement },
  };
  const metadataInput: CppCuteAotRunMetadataV1 = {
    ...metadataBody,
    runMetadataId: `bg.cpp.aot-run-metadata.sha256.${await deriveCppCuteAotRunMetadataHash(metadataBody)}`,
  };
  const metadata = await prepareCppCuteAotRunMetadata(preparedRequest, metadataInput);
  const requestBinding = await prepareCppCuteFrontendRequestBinding(preparedRequest, artifactResource);
  const receiptBody = await createCppCuteAotReceiptBody(metadata, requestBinding, executionEnvironment, artifact);
  const receipt = {
    ...receiptBody,
    receiptId: await deriveCppCuteAotRunnerReceiptId(receiptBody),
  } satisfies CppCuteAotRunnerReceiptV3;
  return {
    profile,
    executionEnvironment,
    artifact,
    artifactResource,
    request: preparedRequest,
    metadata,
    requestBinding,
    sourceSnapshots,
    receipt,
  };
}

export async function createCppCuteAotReceiptBody(
  metadata: PreparedCppCuteAotRunMetadata,
  requestBinding: PreparedCppCuteFrontendRequestBinding,
  executionEnvironment: PreparedCppCuteAotExecutionEnvironment,
  artifact: VerifiedCppCuteFrontendArtifact,
): Promise<CppCuteAotRunnerReceiptBodyV3> {
  const profile = unwrapPreparedCppCuteFrontendProfileFromMetadata(metadata);
  const configured = unwrapPreparedCppCuteAotFrontendProfile(profile).profile;
  const invocationManifestSha256 = await computeCppCuteAotInvocationManifestHash(metadata);
  return {
    schema: CPP_CUTE_AOT_RECEIPT_SCHEMA,
    version: { major: 3, minor: 0 },
    runMetadataId: metadata.runMetadataId,
    requestId: metadata.requestId,
    requestBindingId: requestBinding.bindingId,
    profileHash: profile.profileHash,
    invocation: {
      invocationId: `bg.cpp.aot-invocation.sha256.${invocationManifestSha256}`,
      invocationManifestSha256,
      executionPlanSha256: await computeCppCuteAotExecutionPlanHash(metadata, executionEnvironment),
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
      sourceSetSha256: artifact.sourceSetSha256,
      headerSetSha256: artifact.headerSetSha256,
      inputClosureSha256: artifact.inputClosureSha256,
    },
    output: {
      artifactId: artifact.artifactId,
      artifactHash: artifact.artifactHash,
      transportHash: artifact.transportHash,
      artifactBytesSha256: artifact.artifactBytesSha256,
      artifactByteLength: artifact.artifactByteLength,
      outputManifestSha256: await computeCppCuteAotOutputManifestHash(artifact),
    },
    resources: createResourceAccounting(artifact, configured.extractionLimits),
    outcome: "succeeded",
    exitCode: 0,
  };
}

export function cppCuteAotFixtureSourceBytes(): Uint8Array {
  const bytes = new Uint8Array(100);
  for (let index = 1; index < bytes.length; index += 1) bytes[index] = (index * 31) & 0xff;
  return bytes;
}

async function createRequest(
  profile: PreparedCppCuteFrontendProfile,
  payload: CppCuteFrontendPayloadV3,
  main: CppCuteFrontendPayloadV3["inputs"]["files"][number],
  bytes: Uint8Array,
): Promise<CppCuteFrontendRequestV1> {
  const file = {
    fileId: main.fileId,
    role: "main-source" as const,
    virtualPath: main.virtualPath,
    contentSha256: main.contentSha256,
    byteLength: main.byteLength,
    includeRootId: null,
  };
  const anchor = {
    virtualPath: main.virtualPath,
    beginByte: wire(0),
    endByte: wire(bytes.byteLength),
    tokenSha256: await sha256Hex(bytes),
  };
  const selectedEntryId = payload.outcome.kind === "accepted"
    ? payload.outcome.selectedEntryIds[0]
    : undefined;
  const selectedEntry = payload.entries.find((entry) => entry.entryId === selectedEntryId);
  const entryKind = selectedEntry?.kind === "view-copy"
    ? { kind: "view-copy" as const, declarationKind: "function" as const }
    : selectedEntry?.kind === "logical-gemm-tile"
      ? { kind: "logical-gemm-tile" as const, declarationKind: "function" as const }
      : { kind: "layout" as const, declarationKind: "variable" as const };
  const logicalGemm = selectedEntry?.kind === "logical-gemm-tile";
  const entryBody = {
    requestId: `bg.cpp.entry-request.sha256.${"0".repeat(64)}`,
    ...entryKind,
    anchor,
  };
  const entry: CppCuteFrontendEntryRequestV1 = {
    ...entryBody,
    requestId: await deriveCppCuteFrontendEntryRequestId(entryBody),
  };
  const body: CppCuteFrontendRequestBodyV1 = {
    schema: CPP_CUTE_FRONTEND_REQUEST_SCHEMA,
    version: {
      major: 1,
      minor: logicalGemm ? CPP_CUTE_FRONTEND_REQUEST_LOGICAL_GEMM_TILE_MINOR : 0,
    },
    compilationContractHash: profile.compilationContractHash,
    mainVirtualPath: main.virtualPath,
    files: [file],
    entryRequests: [entry],
    expectedArtifact: {
      schema: "browsergrad.compiler.cpp-cute.frontend-artifact",
      version: {
        major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
        minor: logicalGemm
          ? CPP_CUTE_FRONTEND_ARTIFACT_LOGICAL_GEMM_TILE_MINOR
          : CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
      },
    },
    limits: semanticLimits(profile),
  };
  return { ...body, requestId: `bg.cpp.frontend-request.sha256.${await deriveCppCuteFrontendRequestHash(body)}` };
}

async function createRealSourceBackedArtifact(
  compilationContractHash: string,
  bytes: Uint8Array,
  outcome: "accepted" | "rejected",
  mutatePayload?: (payload: CppCuteFrontendPayloadV3) => void | Promise<void>,
): Promise<CppCuteFrontendArtifactV3> {
  const artifact = await createCppCuteArtifactInput(compilationContractHash);
  const payload = structuredClone(artifact.payload) as CppCuteFrontendPayloadV3;
  const main = payload.inputs.files.find((file) => file.role === "main-source");
  if (main === undefined) throw new Error("fixture lost source file");
  const oldFileId = main.fileId;
  const body = {
    role: "main-source" as const,
    virtualPath: main.virtualPath,
    contentSha256: await sha256Hex(bytes),
    byteLength: encodeWireU64(BigInt(bytes.byteLength)),
    includeRootId: null,
  };
  const newFileId = await deriveCppCuteFrontendSourceFileId(body);
  replaceString(payload, oldFileId, newFileId);
  const rewritten = payload.inputs.files.find((file) => file.fileId === newFileId)!;
  (rewritten as { contentSha256: string }).contentSha256 = body.contentSha256;
  (rewritten as { byteLength: WireU64 }).byteLength = body.byteLength;
  await rebindCppCuteFixtureSourceEntityIds(payload);
  await mutatePayload?.(payload);
  const artifactMinor = payload.entries.some((entry) => entry.kind === "logical-gemm-tile")
    ? CPP_CUTE_FRONTEND_ARTIFACT_LOGICAL_GEMM_TILE_MINOR
    : CPP_CUTE_FRONTEND_ARTIFACT_MINOR;
  const hashes = await computeCppCuteInputHashes(payload);
  (payload.inputs as { sourceSetSha256: string }).sourceSetSha256 = hashes.sourceSetSha256;
  (payload.inputs as { headerSetSha256: string }).headerSetSha256 = hashes.headerSetSha256;
  (payload.inputs as { closureSha256: string }).closureSha256 = hashes.closureSha256;
  for (const [index, pass] of payload.semanticPasses.entries()) {
    (pass as { observedInputClosureSha256: string }).observedInputClosureSha256 =
      await computeCppCuteSemanticPassInputClosureHash(payload, index);
    (pass as { sharedSurfaceSha256: string }).sharedSurfaceSha256 =
      await computeCppCuteSharedSurfaceHash(payload, pass.domain);
  }
  (payload.extraction as { inputClosureSha256: string }).inputClosureSha256 = hashes.closureSha256;
  if (outcome === "rejected") rejectPayload(payload);
  return {
    ...artifact,
    version: { major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR, minor: artifactMinor },
    artifactId: await deriveCppCuteFrontendArtifactId(payload, { minor: artifactMinor }),
    payload,
  };
}

function rejectPayload(payload: CppCuteFrontendPayloadV3): void {
  const blockingDiagnosticId = `bg.cpp.diagnostic.sha256.${"1".repeat(64)}`;
  (payload.diagnostics as unknown as Array<unknown>).push({
    diagnosticId: blockingDiagnosticId,
    phase: "artifact-extraction",
    severity: "error",
    code: "browsergrad.cpp-cute:fixture-rejected",
    renderedMessage: "Fixture rejection for offline-runner coverage.",
    location: { kind: "none" },
    subject: { kind: "compiler" },
    parentDiagnosticId: null,
  });
  (payload.diagnostics as unknown as Array<{ diagnosticId: string }>).sort((a, b) => a.diagnosticId.localeCompare(b.diagnosticId));
  const hostPass = payload.semanticPasses[1]!;
  (hostPass as { status: string }).status = "failed";
  (hostPass as { diagnosticIds: readonly string[] }).diagnosticIds = [blockingDiagnosticId];
  (payload as { outcome: unknown }).outcome = { kind: "rejected", blockingDiagnosticIds: [blockingDiagnosticId] };
}

function unwrapPreparedCppCuteFrontendProfileFromMetadata(metadata: PreparedCppCuteAotRunMetadata): PreparedCppCuteFrontendProfile {
  // Metadata derives and stores the exact request-owned profile; no caller profile is accepted here.
  return unwrapPreparedCppCuteAotRunMetadata(metadata).profile;
}

function semanticLimits(profile: PreparedCppCuteFrontendProfile): CppCuteFrontendRequestLimitsV1 {
  const limits = profile.extractionLimits;
  return Object.fromEntries(Object.entries(limits).filter(([key]) => ![
    "maxWallTimeMs", "maxCpuTimeMs", "maxMemoryBytes", "maxProcesses",
  ].includes(key))) as CppCuteFrontendRequestLimitsV1;
}

function createResourceAccounting(
  artifact: VerifiedCppCuteFrontendArtifact,
  limits: CppCuteFrontendExtractionLimits,
): CppCuteAotReceiptResourcesV3 {
  const payload = unwrapVerifiedCppCuteFrontendArtifact(artifact).envelope.payload;
  const sources = payload.inputs.files.filter((file) => file.owner.kind === "source");
  const headers = payload.inputs.files.filter((file) => file.owner.kind !== "source");
  const sum = (files: typeof payload.inputs.files): bigint => files.reduce((total, file) => total + wireIntegerToBigInt(file.byteLength), 0n);
  return {
    observedInputs: { accountingKind: "observed-exact", values: {
      openedSourceFiles: wire(sources.length), openedSourceBytes: wire(sum(sources)),
      openedHeaderFiles: wire(headers.length), openedHeaderBytes: wire(sum(headers)),
    } },
    processMeasurements: { accountingKind: "observed-exact", values: {
      wallTimeMs: wire(25), cpuTimeMs: wire(20), peakMemoryBytes: wire(65_536), peakProcesses: wire(1),
    } },
    emittedArtifact: { accountingKind: "emitted-artifact-exact", values: {
      macroExpansionFacts: wire(payload.macroExpansions.length),
      templateInstantiationFacts: wire(payload.templateInstantiations.length),
      declarations: wire(payload.declarations.length), types: wire(payload.types.length),
      constants: wire(payload.constants.length),
      layoutFacts: wire(payload.facts.filter((fact) => fact.kind === "affine-layout").length),
      tensorFacts: wire(payload.facts.filter((fact) => fact.kind === "tensor").length),
      operationFacts: wire(payload.facts.filter((fact) => !["affine-layout", "tensor", "target-intrinsic"].includes(fact.kind)).length),
      targetIntrinsicFacts: wire(payload.facts.filter((fact) => fact.kind === "target-intrinsic").length),
      diagnostics: wire(payload.diagnostics.length), outputBytes: artifact.artifactByteLength,
    } },
    enforcedCeilings: { accountingKind: "enforced-upper-bound", values: {
      maxSourceFiles: wire(limits.maxSourceFiles), maxSourceBytes: wire(limits.maxSourceBytes),
      maxHeaderFiles: wire(limits.maxHeaderFiles), maxHeaderBytes: wire(limits.maxHeaderBytes),
      maxIncludeDepth: wire(limits.maxIncludeDepth), maxMacroExpansions: wire(limits.maxMacroExpansions),
      maxPreprocessedTokens: wire(limits.maxPreprocessedTokens), maxAstNodes: wire(limits.maxAstNodes),
      maxConstexprSteps: wire(limits.maxConstexprSteps), maxTemplateInstantiations: wire(limits.maxTemplateInstantiations),
      maxTemplateDepth: wire(limits.maxTemplateDepth), maxDeclarations: wire(limits.maxDeclarations),
      maxTypes: wire(limits.maxTypes), maxConstants: wire(limits.maxConstants), maxLayouts: wire(limits.maxLayouts),
      maxTensors: wire(limits.maxTensors), maxOperations: wire(limits.maxOperations),
      maxTargetIntrinsics: wire(limits.maxTargetIntrinsics), maxDiagnostics: wire(limits.maxDiagnostics),
      maxOutputBytes: wire(limits.maxOutputBytes), maxWallTimeMs: wire(limits.maxWallTimeMs),
      maxCpuTimeMs: wire(limits.maxCpuTimeMs), maxMemoryBytes: wire(limits.maxMemoryBytes),
      maxProcesses: wire(limits.maxProcesses),
    } },
  };
}

function replaceString(value: unknown, target: string, replacement: string): void {
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index += 1) {
      if (value[index] === target) value[index] = replacement;
      else replaceString(value[index], target, replacement);
    }
    return;
  }
  if (typeof value !== "object" || value === null) return;
  for (const [key, entry] of Object.entries(value)) {
    if (entry === target) (value as Record<string, unknown>)[key] = replacement;
    else replaceString(entry, target, replacement);
  }
}
