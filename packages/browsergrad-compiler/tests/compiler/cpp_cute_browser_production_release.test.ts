import { beforeAll, describe, expect, it } from "vitest";
import type {
  VerifiedCppCuteBrowserDistributionApproval,
} from "../../src/cpp_cute_browser_distribution_approval.js";
import {
  cppCuteBrowserExactDistributionConvergenceResourceBytes,
  verifyCppCuteBrowserExactDistributionConvergenceResource,
} from "../../src/cpp_cute_browser_exact_distribution_convergence.js";
import {
  cppCuteBrowserFullDistributionReproducibilityResourceBytes,
  verifyCppCuteBrowserFullDistributionReproducibilityResource,
} from "../../src/cpp_cute_browser_full_distribution_reproducibility.js";
import {
  authorizeCppCuteBrowserProductionBackendExecution,
  type VerifiedCppCuteBrowserProductionBackendExecution,
} from "../../src/cpp_cute_browser_production_backend_execution.js";
import {
  authorizeCppCuteBrowserProductionRelease,
  CppCuteBrowserProductionReleaseError,
  unwrapVerifiedCppCuteBrowserProductionRelease,
} from "../../src/cpp_cute_browser_production_release.js";
import {
  createCurrentPayloadExternallyRootedProducer,
} from "./support/cpp_cute_browser_production_backend_fixtures.js";
import {
  createExternallyRootedDistributionApproval,
} from "./support/cpp_cute_browser_production_release_fixtures.js";

interface ReleaseFixture {
  readonly backend:
    VerifiedCppCuteBrowserProductionBackendExecution;
  readonly approval: VerifiedCppCuteBrowserDistributionApproval;
}

describe("C++/CuTe production release authority", () => {
  let fixture: ReleaseFixture;

  beforeAll(async () => {
    fixture = await createReleaseFixture();
  });

  it("composes backend execution and legal approval without collapsing their authorities", async () => {
    const release = await authorizeCppCuteBrowserProductionRelease(
      fixture.backend,
      fixture.approval,
    );

    expect(release).toMatchObject({
      authority: "externally-approved-browser-cpp-cute-release",
      backendExecutionAuthorityId:
        fixture.backend.backendExecutionAuthorityId,
      producerEvidenceId: fixture.backend.producerEvidenceId,
      fullDistributionReproducibilityId:
        fixture.backend.fullDistributionReproducibilityId,
      fullDistributionResourceSha256:
        fixture.backend.fullDistributionResourceSha256,
      exactDistributionConvergenceMatrixId:
        fixture.backend.exactDistributionConvergenceMatrixId,
      buildSubjectId: fixture.backend.buildSubjectId,
      buildInputLockId: fixture.backend.buildInputLockId,
      distributionApprovalEvidenceId:
        fixture.approval.approvalEvidenceId,
      distributionApprovalPolicyId: fixture.approval.policyId,
      reviewerId: fixture.approval.reviewerId,
      reviewerKeyId: fixture.approval.keyId,
      distributionReviewSubjectId: fixture.approval.reviewSubjectId,
      headerDistributionResourceSha256:
        fixture.approval.headerDistributionResourceSha256,
      headerDistributionReproducibilityId:
        fixture.approval.headerDistributionReproducibilityId,
      headerDistributionOutputVerificationId:
        fixture.approval.headerDistributionOutputVerificationId,
      externallyRootedProducerTrusted: true,
      fullDistributedOutputSetReproducible: true,
      exactPrivateDistributionTreeVerified: true,
      exactNineCaseBrowserWorkerCompilationObserved: true,
      exactCandidatesAuthorizedThroughSharedSeam: true,
      cpuReferenceConvergenceObservedForEveryCase: true,
      requiredRealWebGpuConvergenceObservedForEveryCase: true,
      completeDestinationBitComparisonPassedForEveryCase: true,
      nonzeroOffsetCanariesPreservedForEveryCase: true,
      workerExecutionObserved: true,
      loweringAuthorityMinted: true,
      backendExecutionObserved: true,
      backendExecutionAuthorityMinted: true,
      externalDistributedFileLicenseMapReviewed: true,
      exactPackageNoticeSetReviewed: true,
      exactCudaRedistributionIndexReviewed: true,
      exactUpstreamLicenseEvidenceReviewed: true,
      licenseReviewComplete: true,
      distributionAuthorized: true,
      finalReleaseAuthorityMinted: true,
      releaseReady: true,
    });
    expect(release.releaseAuthorityId).toMatch(
      /^bg\.cpp\.browser-production-release\.sha256\.[0-9a-f]{64}$/u,
    );
    expect(unwrapVerifiedCppCuteBrowserProductionRelease(release))
      .toEqual({
        backendExecution: fixture.backend,
        distributionApproval: fixture.approval,
      });
  });

  it("rejects structural copies of either prerequisite and result", async () => {
    await expect(authorizeCppCuteBrowserProductionRelease(
      { ...fixture.backend },
      fixture.approval,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-RELEASE-UNVERIFIED",
      path: "$.backendExecution",
    });
    await expect(authorizeCppCuteBrowserProductionRelease(
      fixture.backend,
      { ...fixture.approval },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-RELEASE-UNVERIFIED",
      path: "$.distributionApproval",
    });

    const release = await authorizeCppCuteBrowserProductionRelease(
      fixture.backend,
      fixture.approval,
    );
    expect(() =>
      unwrapVerifiedCppCuteBrowserProductionRelease({ ...release }),
    ).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-RELEASE-UNVERIFIED",
      path: "$",
    }));
  });

  it("checks cancellation and hostile options before release issuance", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(authorizeCppCuteBrowserProductionRelease(
      fixture.backend,
      fixture.approval,
      { signal: controller.signal },
    )).rejects.toBeInstanceOf(CppCuteBrowserProductionReleaseError);

    const hostileOptions = new Proxy({}, {
      ownKeys: () => {
        throw new Error("hostile options trap");
      },
    });
    await expect(authorizeCppCuteBrowserProductionRelease(
      fixture.backend,
      fixture.approval,
      hostileOptions as never,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-RELEASE-BINDING",
      path: "$.options",
    });
  });
});

async function createReleaseFixture(): Promise<ReleaseFixture> {
  const fullDistribution =
    await verifyCppCuteBrowserFullDistributionReproducibilityResource(
      cppCuteBrowserFullDistributionReproducibilityResourceBytes(),
    );
  const [producer, convergence, approval] = await Promise.all([
    createCurrentPayloadExternallyRootedProducer(fullDistribution),
    verifyCppCuteBrowserExactDistributionConvergenceResource(
      cppCuteBrowserExactDistributionConvergenceResourceBytes(),
    ),
    createExternallyRootedDistributionApproval(),
  ]);
  return {
    backend: await authorizeCppCuteBrowserProductionBackendExecution(
      producer,
      fullDistribution,
      convergence,
    ),
    approval,
  };
}
