import { describe, expect, it } from "vitest";
import {
  CPP_CUTE_BROWSER_REAL_COMPILE_BASELINE_CASE_IDS,
  cppCuteBrowserRealCompileCase,
} from "../../src/cpp_cute_browser_real_compile_cases.js";
import {
  CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_MATRIX_SCHEMA,
  CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_OBSERVATION_SCHEMA,
  isRetryableCppCuteBrowserExactDistributionFailure,
  parseCppCuteBrowserExactDistributionConvergenceArguments,
  prepareCppCuteBrowserExactDistributionConvergenceMatrix,
  type CppCuteBrowserExactDistributionConvergencePreflight,
} from "./cpp_cute_browser_exact_distribution_convergence.mjs";

const sourceRevision = "a".repeat(40);
const preflight = {
  schema:
    "browsergrad.compiler.cpp-cute.browser-exact-distribution-convergence-inputs",
  version: 1,
  authority: "host-preflight-exact-private-distribution-only",
  controls: {},
  assets: [],
  distribution: {
    reproducibilityId:
      `bg.cpp.browser-full-distribution-reproducibility.sha256.${"1".repeat(64)}`,
    buildSubjectId:
      `bg.cpp.browser-build-subject.sha256.${"2".repeat(64)}`,
  },
  producer: {
    producerEvidenceId:
      `bg.cpp.browser-build-producer.sha256.${"3".repeat(64)}`,
  },
  claims: {},
} as unknown as CppCuteBrowserExactDistributionConvergencePreflight;

describe("exact distribution browser convergence harness", () => {
  it("parses only the closed absolute-path invocation", () => {
    const parsed =
      parseCppCuteBrowserExactDistributionConvergenceArguments([
        "--distribution-root=/private/distribution",
        "--profile=/private/profile.json",
        "--producer-policy=/private/policy.json",
        "--producer-trust-store=/private/trust-store.json",
        "--checkpoint-directory=/private/checkpoints",
        "--evidence-output=/private/evidence.json",
        `--source-revision=${sourceRevision}`,
      ]);
    expect(parsed).toEqual({
      distributionRoot: "/private/distribution",
      profilePath: "/private/profile.json",
      producerPolicyPath: "/private/policy.json",
      producerTrustStorePath: "/private/trust-store.json",
      checkpointDirectory: "/private/checkpoints",
      evidenceOutput: "/private/evidence.json",
      sourceRevision,
      preflightOnly: false,
    });
    expect(() =>
      parseCppCuteBrowserExactDistributionConvergenceArguments([
        "--distribution-root=relative",
        "--profile=/profile.json",
        "--producer-policy=/policy.json",
        "--producer-trust-store=/trust-store.json",
        "--preflight-only",
      ])
    ).toThrowError(expect.objectContaining({
      code:
        "BG-COMPILER-CPP-CUTE-BROWSER-EXACT-DISTRIBUTION-CONVERGENCE",
      path: "$.distributionRoot",
    }));
    expect(() =>
      parseCppCuteBrowserExactDistributionConvergenceArguments([
        "--distribution-root=/distribution",
        "--profile=/profile.json",
        "--producer-policy=/policy.json",
        "--producer-trust-store=/trust-store.json",
        "--unknown=/value",
      ])
    ).toThrowError(expect.objectContaining({
      path: "$.argv[4]",
    }));
    expect(() =>
      parseCppCuteBrowserExactDistributionConvergenceArguments([
        "--distribution-root=/distribution",
        "--profile=/profile.json",
        "--producer-policy=/policy.json",
        "--producer-trust-store=/trust-store.json",
        "--checkpoint-directory=/checkpoints",
        "--preflight-only",
      ])
    ).toThrowError(expect.objectContaining({
      path: "$.argv",
    }));
  });

  it("retries only pre-evidence browser transport disconnects", () => {
    expect(isRetryableCppCuteBrowserExactDistributionFailure(
      "Browser connection was closed while running tests",
    )).toBe(true);
    expect(isRetryableCppCuteBrowserExactDistributionFailure(
      "rpc is closed, cannot call \"createTesters\"",
    )).toBe(true);
    expect(isRetryableCppCuteBrowserExactDistributionFailure(
      "TypeError: compiler semantic mismatch",
    )).toBe(false);
    expect(isRetryableCppCuteBrowserExactDistributionFailure(
      `BROWSERGRAD_CPP_CUTE_EXACT_DISTRIBUTION_CONVERGENCE_EVIDENCE={}\n` +
      "Browser connection was closed while running tests",
    )).toBe(false);
  });

  it("closes the exact eight-case matrix without widening authority", () => {
    const observations =
      CPP_CUTE_BROWSER_REAL_COMPILE_BASELINE_CASE_IDS.map(observation);
    const matrix =
      prepareCppCuteBrowserExactDistributionConvergenceMatrix(
        observations,
        preflight,
        sourceRevision,
      );
    expect(matrix).toMatchObject({
      schema:
        CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_MATRIX_SCHEMA,
      version: 1,
      authority:
        "local-engineering-exact-payload-cpu-webgpu-observation-only",
      sourceRevision,
      caseCount: 8,
      webgpu: {
        required: true,
        actualExecutionObservedForEveryCase: true,
        deviceProfileCount: 1,
      },
      claims: {
        exactEightCaseBrowserWorkerCompilationObserved: true,
        exactCandidatesAuthorizedThroughSharedSeam: true,
        cpuReferenceConvergenceObservedForEveryCase: true,
        requiredRealWebGpuConvergenceObservedForEveryCase: true,
        externalProducerTrusted: false,
        licenseReviewComplete: false,
        distributionAuthorized: false,
        backendExecutionAuthorityMinted: false,
        releaseReady: false,
      },
    });
    expect(matrix.matrixId).toMatch(
      /^bg\.cpp\.browser-exact-distribution-convergence\.sha256\.[0-9a-f]{64}$/u,
    );
  });

  it("rejects widened claims and reused opaque lineage IDs", () => {
    const observations =
      CPP_CUTE_BROWSER_REAL_COMPILE_BASELINE_CASE_IDS.map(observation);
    const widened = structuredClone(observations);
    widened[0]!.claims.externalProducerTrusted = true;
    expect(() =>
      prepareCppCuteBrowserExactDistributionConvergenceMatrix(
        widened,
        preflight,
        sourceRevision,
      )
    ).toThrowError(expect.objectContaining({
      path: "$.observations[0]",
    }));

    const reused = structuredClone(observations);
    reused[1]!.execution.candidateId =
      reused[0]!.execution.candidateId;
    expect(() =>
      prepareCppCuteBrowserExactDistributionConvergenceMatrix(
        reused,
        preflight,
        sourceRevision,
      )
    ).toThrowError(expect.objectContaining({
      path: "$.observations",
      message: expect.stringContaining("candidateId"),
    }));

    const staleRevision = structuredClone(observations);
    staleRevision[0]!.sourceRevision = "b".repeat(40);
    expect(() =>
      prepareCppCuteBrowserExactDistributionConvergenceMatrix(
        staleRevision,
        preflight,
        sourceRevision,
      )
    ).toThrowError(expect.objectContaining({
      path: "$.observations[0]",
    }));
  });
});

function observation(
  caseId: (typeof CPP_CUTE_BROWSER_REAL_COMPILE_BASELINE_CASE_IDS)[number],
  index: number,
) {
  const compileCase = cppCuteBrowserRealCompileCase(caseId);
  const digest = (value: number) =>
    (value.toString(16).padStart(2, "0").repeat(32)).slice(0, 64);
  return {
    schema:
      CPP_CUTE_BROWSER_EXACT_DISTRIBUTION_CONVERGENCE_OBSERVATION_SCHEMA,
    version: 1,
    evidenceId:
      `bg.cpp.browser-exact-distribution-case-convergence.sha256.${digest(index + 1)}`,
    caseId,
    sourceRevision,
    source: {
      sourceSha256: compileCase.sourceSha256,
      dtype: compileCase.dtype,
      coordinateRank: compileCase.coordinateRank,
    },
    distribution: {
      reproducibilityId:
        preflight.distribution.reproducibilityId,
      buildSubjectId: preflight.distribution.buildSubjectId,
    },
    producer: {
      producerEvidenceId: preflight.producer.producerEvidenceId,
    },
    execution: {
      candidateId: `candidate-${index}`,
      artifactId: `artifact-${index}`,
      authorizationId: `authorization-${index}`,
      executionEvidenceId: `execution-${index}`,
      layoutSemanticHash: digest(index + 20),
      kernelSemanticHash: digest(index + 40),
      cpuDestinationHash: digest(index + 60),
      webGpuDestinationHash: digest(index + 60),
      browserWorkerCompiled: true,
      localSemanticAuthorizationMinted: true,
      cpuReferenceExecuted: true,
      actualWebGpuExecuted: true,
      completeDestinationBitComparisonPassed: true,
      nonzeroOffsetCanariesPreserved: true,
    },
    webgpu: {
      deviceProfileHash: "9".repeat(64),
    },
    claims: {
      externalProducerTrusted: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      backendExecutionAuthorityMinted: false,
      releaseReady: false,
    },
  };
}
