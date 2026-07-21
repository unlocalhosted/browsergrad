import { describe, expect, it } from "vitest";
import { EXECUTION_ENVIRONMENT_SCHEMA, EXECUTION_EVIDENCE_SCHEMA } from
  "../../../../test-support/webgpu-evidence";
import {
  CPP_CUTE_VIEW_COPY_WEBGPU_BACKEND_ID,
  CPP_CUTE_VIEW_COPY_WEBGPU_CAPABILITY_ID,
  CPP_CUTE_VIEW_COPY_WEBGPU_CASE_ID,
  CPP_CUTE_VIEW_COPY_WEBGPU_COMPARISON_POLICY_ID,
  CPP_CUTE_VIEW_COPY_WEBGPU_RANK3_CASE_ID,
  CPP_CUTE_VIEW_COPY_WEBGPU_SUITE_ID,
  finalizeCppCuteViewCopyWebGpuEvidence,
  type CppCuteViewCopyWebGpuEvidenceExpectation,
  type CppCuteViewCopyWebGpuTerminalEvidence,
} from "../../tests-browser/cpp_cute_view_copy_webgpu_evidence";

const hash = (digit: string) => digit.repeat(64);
const PRODUCERS = Object.freeze({ compiler: "0.2.0", kernels: "0.2.0" });
const EXPECTED: CppCuteViewCopyWebGpuEvidenceExpectation = Object.freeze({
  expectedRequired: true,
  expectedFixtureSchema: "browsergrad.compiler.cpp-cute-browser-view-copy-convergence@1",
  expectedCaseId: CPP_CUTE_VIEW_COPY_WEBGPU_CASE_ID,
  expectedFixtureArtifactHash: hash("1"),
  expectedInputHash: hash("2"),
  expectedDestinationHash: hash("3"),
  expectedEnvironmentId: hash("4"),
  expectedSourceRevision: "a".repeat(40),
  expectedProducerVersions: PRODUCERS,
});

describe("CuTe fixture-payload WebGPU terminal evidence", () => {
  it("accepts only a complete actual-WebGPU pass bound to exact hashes and canaries", () => {
    const record = passingRecord();
    expect(finalizeCppCuteViewCopyWebGpuEvidence(record, EXPECTED))
      .toMatchObject({ actualWebGpuExecution: true, case: {
        completeDestinationBitComparisonPassed: true,
        nonzeroOffsetCanariesPreserved: true,
      } });
    const rank3Expected = Object.freeze({
      ...EXPECTED,
      expectedCaseId: CPP_CUTE_VIEW_COPY_WEBGPU_RANK3_CASE_ID,
    });
    const rank3 = passingRecord(rank3Expected);
    expect(finalizeCppCuteViewCopyWebGpuEvidence(rank3, rank3Expected))
      .toMatchObject({ fixtureCaseId: CPP_CUTE_VIEW_COPY_WEBGPU_RANK3_CASE_ID });
    expect(() => finalizeCppCuteViewCopyWebGpuEvidence(rank3, EXPECTED))
      .toThrow(/fixture identity differs/u);

    expect(() => finalizeCppCuteViewCopyWebGpuEvidence({
      ...record,
      actualWebGpuExecution: false,
    }, EXPECTED)).toThrow(/actual WebGPU execution/u);
    expect(() => finalizeCppCuteViewCopyWebGpuEvidence({
      ...record,
      case: { ...record.case!, actualDestinationHash: hash("5") },
    }, EXPECTED)).toThrow(/exact payload hashes and canary comparison/u);
    expect(() => finalizeCppCuteViewCopyWebGpuEvidence({
      ...record,
      case: { ...record.case!, cpuSemanticSpecializationHash: "short" },
    }, EXPECTED)).toThrow(/full SHA-256 digests/u);
    expect(() => finalizeCppCuteViewCopyWebGpuEvidence({
      ...record,
      case: { ...record.case!, nonzeroOffsetCanariesPreserved: false },
    }, EXPECTED)).toThrow(/exact payload hashes and canary comparison/u);
    expect(() => finalizeCppCuteViewCopyWebGpuEvidence({
      ...record,
      case: { ...record.case!, gpuSemanticSpecializationHash: hash("d") },
    }, EXPECTED)).toThrow(/exact payload hashes and canary comparison/u);
    expect(() => finalizeCppCuteViewCopyWebGpuEvidence({
      ...record,
      case: { ...record.case!, caseId: CPP_CUTE_VIEW_COPY_WEBGPU_RANK3_CASE_ID },
    }, EXPECTED)).toThrow(/exact payload hashes and canary comparison/u);
    expect(() => finalizeCppCuteViewCopyWebGpuEvidence({
      ...record,
      uncapturedErrors: ["late validation error"],
    }, EXPECTED)).toThrow(/clean complete observation/u);
  });
});

function passingRecord(
  expected: CppCuteViewCopyWebGpuEvidenceExpectation = EXPECTED,
): CppCuteViewCopyWebGpuTerminalEvidence {
  return {
    schema: EXECUTION_EVIDENCE_SCHEMA,
    kind: "terminal",
    suiteId: CPP_CUTE_VIEW_COPY_WEBGPU_SUITE_ID,
    required: true,
    evidence: {
      capabilityId: CPP_CUTE_VIEW_COPY_WEBGPU_CAPABILITY_ID,
      artifactHash: expected.expectedFixtureArtifactHash,
      backendId: CPP_CUTE_VIEW_COPY_WEBGPU_BACKEND_ID,
      environmentId: expected.expectedEnvironmentId,
      producerVersions: PRODUCERS,
      sourceRevision: expected.expectedSourceRevision,
      deviceProfileHash: hash("6"),
      recordedAt: "2026-07-21T00:00:00.000Z",
      outcome: "passed",
      comparisonPolicyId: CPP_CUTE_VIEW_COPY_WEBGPU_COMPARISON_POLICY_ID,
      diagnosticCodes: [],
    },
    environment: {
      schema: EXECUTION_ENVIRONMENT_SCHEMA,
      acquisition: "navigator.gpu.requestAdapter/requestDevice",
      userAgent: "test",
      platform: "test",
      adapter: {},
      adapterSupportedFeatures: [],
      negotiatedDeviceFeatures: [],
      negotiatedDeviceLimits: {},
    },
    artifactHashKind: "pinned-cpp-cute-view-copy-convergence-fixture",
    fixtureSchema: expected.expectedFixtureSchema,
    fixtureCaseId: expected.expectedCaseId,
    inputHash: expected.expectedInputHash,
    expectedDestinationHash: expected.expectedDestinationHash,
    preparedBackendArtifactHash: hash("7"),
    productionBrowserCompileObserved: false,
    actualWebGpuExecution: true,
    backendExecutionAuthorizationMinted: false,
    cudaLiteRunnerUsed: false,
    case: {
      caseId: expected.expectedCaseId,
      fixtureArtifactHash: expected.expectedFixtureArtifactHash,
      inputHash: expected.expectedInputHash,
      expectedDestinationHash: expected.expectedDestinationHash,
      actualDestinationHash: expected.expectedDestinationHash,
      layoutSemanticHash: hash("8"),
      kernelSemanticHash: hash("9"),
      operationId: "bg.entity.kernel-operation.test",
      cpuSemanticSpecializationHash: hash("a"),
      gpuSemanticSpecializationHash: hash("a"),
      preparedBackendArtifactHash: hash("7"),
      wgslModuleHash: hash("b"),
      backendSpecializationHash: hash("c"),
      backendProfile: "webgpu-core",
      backendVersion: "1",
      deviceProfileHash: hash("6"),
      completeDestinationBitComparisonPassed: true,
      nonzeroOffsetCanariesPreserved: true,
    },
    uncapturedErrors: [],
  };
}
