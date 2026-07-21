import {
  canonicalizeJson,
  deepFreezeJson,
  parseWireJson,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  EXECUTION_EVIDENCE_SCHEMA,
  validateTerminalExecutionEvidence,
} from "../../../test-support/webgpu-evidence";
import type { EvidenceEnvironment } from "./semantic_view_copy_bindings_evidence";

export const CPP_CUTE_VIEW_COPY_WEBGPU_SUITE_ID =
  "browsergrad.compiler.cpp-cute-view-copy-fixture.webgpu-convergence@1";
export const CPP_CUTE_VIEW_COPY_WEBGPU_CAPABILITY_ID =
  "browsergrad.compiler.cpp-cute-view-copy.canonical-fixture-payload";
export const CPP_CUTE_VIEW_COPY_WEBGPU_BACKEND_ID =
  "browsergrad.backend.webgpu.core";
export const CPP_CUTE_VIEW_COPY_WEBGPU_COMPARISON_POLICY_ID =
  "browsergrad.comparison.bit-exact-u32-complete-destination-with-canaries.v1";
export const CPP_CUTE_VIEW_COPY_WEBGPU_RANK2_CASE_ID =
  "canonical-rank2-cute-view-copy-payload";
export const CPP_CUTE_VIEW_COPY_WEBGPU_RANK3_CASE_ID =
  "canonical-rank3-cute-view-copy-payload";
export const CPP_CUTE_VIEW_COPY_WEBGPU_CASE_ID =
  CPP_CUTE_VIEW_COPY_WEBGPU_RANK2_CASE_ID;

export const CPP_CUTE_VIEW_COPY_WEBGPU_TERMINAL_EXPECTATION = Object.freeze({
  suiteId: CPP_CUTE_VIEW_COPY_WEBGPU_SUITE_ID,
  capabilityId: CPP_CUTE_VIEW_COPY_WEBGPU_CAPABILITY_ID,
  backendId: CPP_CUTE_VIEW_COPY_WEBGPU_BACKEND_ID,
  comparisonPolicyId: CPP_CUTE_VIEW_COPY_WEBGPU_COMPARISON_POLICY_ID,
  requireDeviceProfile: true,
});

export interface CppCuteViewCopyWebGpuCaseEvidence extends JsonObject {
  readonly caseId: string;
  readonly fixtureArtifactHash: string;
  readonly inputHash: string;
  readonly expectedDestinationHash: string;
  readonly actualDestinationHash: string;
  readonly layoutSemanticHash: string;
  readonly kernelSemanticHash: string;
  readonly operationId: string;
  readonly cpuSemanticSpecializationHash: string;
  readonly gpuSemanticSpecializationHash: string;
  readonly preparedBackendArtifactHash: string;
  readonly wgslModuleHash: string;
  readonly backendSpecializationHash: string;
  readonly backendProfile: string;
  readonly backendVersion: string;
  readonly deviceProfileHash: string;
  readonly completeDestinationBitComparisonPassed: boolean;
  readonly nonzeroOffsetCanariesPreserved: boolean;
}

export interface CppCuteViewCopyWebGpuTerminalEvidence extends JsonObject {
  readonly schema: typeof EXECUTION_EVIDENCE_SCHEMA;
  readonly kind: "terminal";
  readonly suiteId: typeof CPP_CUTE_VIEW_COPY_WEBGPU_SUITE_ID;
  readonly required: boolean;
  readonly evidence: JsonObject;
  readonly environment: EvidenceEnvironment;
  readonly artifactHashKind: "pinned-cpp-cute-view-copy-convergence-fixture";
  readonly fixtureSchema: string;
  readonly fixtureCaseId: string;
  readonly inputHash: string;
  readonly expectedDestinationHash: string;
  readonly preparedBackendArtifactHash?: string;
  readonly productionBrowserCompileObserved: false;
  readonly actualWebGpuExecution: boolean;
  readonly backendExecutionAuthorizationMinted: false;
  readonly cudaLiteRunnerUsed: false;
  readonly case?: CppCuteViewCopyWebGpuCaseEvidence;
  readonly uncapturedErrors: readonly string[];
  readonly error?: JsonObject;
}

export interface CppCuteViewCopyWebGpuEvidenceExpectation {
  readonly expectedRequired: boolean;
  readonly expectedFixtureSchema: string;
  readonly expectedCaseId: string;
  readonly expectedFixtureArtifactHash: string;
  readonly expectedInputHash: string;
  readonly expectedDestinationHash: string;
  readonly expectedEnvironmentId: string;
  readonly expectedSourceRevision: string;
  readonly expectedProducerVersions: JsonObject;
}

export function finalizeCppCuteViewCopyWebGpuEvidence(
  input: CppCuteViewCopyWebGpuTerminalEvidence,
  expected: CppCuteViewCopyWebGpuEvidenceExpectation,
): CppCuteViewCopyWebGpuTerminalEvidence {
  const record = deepFreezeJson(
    parseWireJson(canonicalizeJson(input)),
  ) as CppCuteViewCopyWebGpuTerminalEvidence;
  validateCppCuteViewCopyWebGpuEvidence(record, expected);
  return record;
}

export function validateCppCuteViewCopyWebGpuEvidence(
  record: CppCuteViewCopyWebGpuTerminalEvidence,
  expected: CppCuteViewCopyWebGpuEvidenceExpectation,
): void {
  validateTerminalExecutionEvidence(
    record,
    CPP_CUTE_VIEW_COPY_WEBGPU_TERMINAL_EXPECTATION,
  );
  if (record.required !== expected.expectedRequired) {
    invalid("required flag differs from the evidence lane mode");
  }
  if (
    record.artifactHashKind !== "pinned-cpp-cute-view-copy-convergence-fixture"
    || record.fixtureSchema !== expected.expectedFixtureSchema
    || record.fixtureCaseId !== expected.expectedCaseId
  ) {
    invalid("fixture identity differs from the pinned convergence payload");
  }
  const evidence = record.evidence;
  if (
    evidence.artifactHash !== expected.expectedFixtureArtifactHash
    || record.inputHash !== expected.expectedInputHash
    || record.expectedDestinationHash !== expected.expectedDestinationHash
    || evidence.environmentId !== expected.expectedEnvironmentId
    || evidence.sourceRevision !== expected.expectedSourceRevision
    || !sameJson(evidence.producerVersions, expected.expectedProducerVersions)
  ) {
    invalid("terminal identities differ from the exact producing fixture and environment");
  }
  if (
    record.productionBrowserCompileObserved !== false
    || record.backendExecutionAuthorizationMinted !== false
    || record.cudaLiteRunnerUsed !== false
  ) {
    invalid("fixture evidence cannot broaden its producer or authorization claims");
  }
  if (
    !Array.isArray(record.uncapturedErrors)
    || record.uncapturedErrors.some((message) => typeof message !== "string" || message.length === 0)
  ) {
    invalid("uncapturedErrors must contain nonempty strings");
  }

  if (evidence.outcome !== "passed") return;
  const observation = record.case;
  if (
    record.actualWebGpuExecution !== true
    || record.preparedBackendArtifactHash === undefined
    || observation === undefined
    || record.error !== undefined
    || record.uncapturedErrors.length !== 0
  ) {
    invalid("passed evidence requires actual WebGPU execution and a clean complete observation");
  }
  const hashes = [
    observation.fixtureArtifactHash,
    observation.inputHash,
    observation.expectedDestinationHash,
    observation.actualDestinationHash,
    observation.layoutSemanticHash,
    observation.kernelSemanticHash,
    observation.cpuSemanticSpecializationHash,
    observation.gpuSemanticSpecializationHash,
    observation.preparedBackendArtifactHash,
    observation.wgslModuleHash,
    observation.backendSpecializationHash,
    observation.deviceProfileHash,
  ];
  if (hashes.some((value) => !digest(value))) {
    invalid("passed case hashes must be full SHA-256 digests");
  }
  if (
    observation.caseId !== expected.expectedCaseId
    || observation.fixtureArtifactHash !== expected.expectedFixtureArtifactHash
    || observation.inputHash !== expected.expectedInputHash
    || observation.expectedDestinationHash !== expected.expectedDestinationHash
    || observation.actualDestinationHash !== expected.expectedDestinationHash
    || observation.preparedBackendArtifactHash !== record.preparedBackendArtifactHash
    || observation.deviceProfileHash !== evidence.deviceProfileHash
    || observation.cpuSemanticSpecializationHash !==
      observation.gpuSemanticSpecializationHash
    || observation.completeDestinationBitComparisonPassed !== true
    || observation.nonzeroOffsetCanariesPreserved !== true
  ) {
    invalid("passed case does not prove the exact payload hashes and canary comparison");
  }
}

function digest(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{64}$/u.test(value);
}

function sameJson(left: unknown, right: unknown): boolean {
  return canonicalizeJson(left as JsonObject) === canonicalizeJson(right as JsonObject);
}

function invalid(message: string): never {
  throw new Error(`invalid CuTe view-copy WebGPU evidence: ${message}`);
}
