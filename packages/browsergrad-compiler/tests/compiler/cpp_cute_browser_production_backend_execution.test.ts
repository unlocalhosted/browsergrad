import { beforeAll, describe, expect, it } from "vitest";
import {
  cppCuteBrowserExactDistributionConvergenceResourceBytes,
  verifyCppCuteBrowserExactDistributionConvergenceResource,
  type VerifiedCppCuteBrowserExactDistributionConvergence,
} from "../../src/cpp_cute_browser_exact_distribution_convergence.js";
import {
  cppCuteBrowserFullDistributionReproducibilityResourceBytes,
  verifyCppCuteBrowserFullDistributionReproducibilityResource,
  type VerifiedCppCuteBrowserFullDistributionReproducibility,
} from "../../src/cpp_cute_browser_full_distribution_reproducibility.js";
import type {
  VerifiedCppCuteBrowserBuildProducer,
} from "../../src/cpp_cute_browser_producer_trust.js";
import {
  authorizeCppCuteBrowserProductionBackendExecution,
  CppCuteBrowserProductionBackendExecutionError,
  unwrapVerifiedCppCuteBrowserProductionBackendExecution,
} from "../../src/cpp_cute_browser_production_backend_execution.js";
import {
  createCurrentPayloadExternallyRootedProducer,
} from "./support/cpp_cute_browser_production_backend_fixtures.js";

interface BackendFixture {
  readonly producer: VerifiedCppCuteBrowserBuildProducer;
  readonly fullDistribution:
    VerifiedCppCuteBrowserFullDistributionReproducibility;
  readonly convergence:
    VerifiedCppCuteBrowserExactDistributionConvergence;
}

describe("C++/CuTe production backend execution authority", () => {
  let fixture: BackendFixture;

  beforeAll(async () => {
    fixture = await createBackendFixture();
  });

  it("re-admits the exact reproducible build under external trust and exact execution evidence", async () => {
    const backend =
      await authorizeCppCuteBrowserProductionBackendExecution(
        fixture.producer,
        fixture.fullDistribution,
        fixture.convergence,
      );

    expect(backend).toMatchObject({
      authority:
        "externally-trusted-browser-exact-payload-backend-execution",
      producerEvidenceId: fixture.producer.producerEvidenceId,
      producerPolicyId: fixture.producer.policyId,
      fullDistributionReproducibilityId:
        fixture.fullDistribution.reproducibilityId,
      fullDistributionResourceSha256:
        fixture.fullDistribution.resourceSha256,
      exactDistributionConvergenceMatrixId: fixture.convergence.matrixId,
      exactDistributionConvergenceResourceSha256:
        fixture.convergence.resourceSha256,
      buildSubjectId: fixture.producer.buildSubjectId,
      buildSubjectSha256: fixture.producer.buildSubjectSha256,
      buildInputLockId: fixture.fullDistribution.buildInputLockId,
      buildInputLockResourceSha256:
        fixture.fullDistribution.buildInputLockResourceSha256,
      producerProfileHash: fixture.producer.profileHash,
      producerAssetManifestId: fixture.producer.manifestId,
      producerAssetSetSha256: fixture.producer.assetSetSha256,
      executionProfileHash:
        fixture.fullDistribution.deterministicMetadata.profileHash,
      executionAssetManifestId:
        fixture.fullDistribution.deterministicMetadata.assetManifestId,
      executionAssetSetSha256:
        fixture.fullDistribution.deterministicMetadata.assetSetSha256,
      workerBundleSha256: fixture.producer.workerBundleSha256,
      webGpuDeviceProfileHash:
        fixture.convergence.webgpu.deviceProfileHashes[0],
      exactCaseCount: 8,
      externallyRootedProducerTrusted: true,
      fullDistributedOutputSetReproducible: true,
      exactPrivateDistributionTreeVerified: true,
      exactEightCaseBrowserWorkerCompilationObserved: true,
      exactCandidatesAuthorizedThroughSharedSeam: true,
      cpuReferenceConvergenceObservedForEveryCase: true,
      requiredRealWebGpuConvergenceObservedForEveryCase: true,
      completeDestinationBitComparisonPassedForEveryCase: true,
      nonzeroOffsetCanariesPreservedForEveryCase: true,
      workerExecutionObserved: true,
      loweringAuthorityMinted: true,
      backendExecutionObserved: true,
      backendExecutionAuthorityMinted: true,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      releaseReady: false,
    });
    expect(backend.backendExecutionAuthorityId).toMatch(
      /^bg\.cpp\.browser-production-backend-execution\.sha256\.[0-9a-f]{64}$/u,
    );
    expect(
      unwrapVerifiedCppCuteBrowserProductionBackendExecution(backend),
    ).toEqual({
      producer: fixture.producer,
      fullDistribution: fixture.fullDistribution,
      convergence: fixture.convergence,
    });
  });

  it("rejects structural copies of every opaque prerequisite and result", async () => {
    const cases = [
      {
        path: "$.producer",
        producer: { ...fixture.producer },
        fullDistribution: fixture.fullDistribution,
        convergence: fixture.convergence,
      },
      {
        path: "$.fullDistribution",
        producer: fixture.producer,
        fullDistribution: { ...fixture.fullDistribution },
        convergence: fixture.convergence,
      },
      {
        path: "$.convergence",
        producer: fixture.producer,
        fullDistribution: fixture.fullDistribution,
        convergence: { ...fixture.convergence },
      },
    ];
    for (const entry of cases) {
      await expect(
        authorizeCppCuteBrowserProductionBackendExecution(
          entry.producer as VerifiedCppCuteBrowserBuildProducer,
          entry.fullDistribution as
            VerifiedCppCuteBrowserFullDistributionReproducibility,
          entry.convergence as
            VerifiedCppCuteBrowserExactDistributionConvergence,
        ),
      ).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-BACKEND-UNVERIFIED",
        path: entry.path,
      });
    }

    const backend =
      await authorizeCppCuteBrowserProductionBackendExecution(
        fixture.producer,
        fixture.fullDistribution,
        fixture.convergence,
      );
    expect(() =>
      unwrapVerifiedCppCuteBrowserProductionBackendExecution({
        ...backend,
      }),
    ).toThrowError(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-BACKEND-UNVERIFIED",
      path: "$",
    }));
  });

  it("checks cancellation and hostile options before minting authority", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(
      authorizeCppCuteBrowserProductionBackendExecution(
        fixture.producer,
        fixture.fullDistribution,
        fixture.convergence,
        { signal: controller.signal },
      ),
    ).rejects.toBeInstanceOf(
      CppCuteBrowserProductionBackendExecutionError,
    );

    const hostileOptions = new Proxy({}, {
      ownKeys: () => {
        throw new Error("hostile options trap");
      },
    });
    await expect(
      authorizeCppCuteBrowserProductionBackendExecution(
        fixture.producer,
        fixture.fullDistribution,
        fixture.convergence,
        hostileOptions as never,
      ),
    ).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-PRODUCTION-BACKEND-BINDING",
      path: "$.options",
    });
  });
});

async function createBackendFixture(): Promise<BackendFixture> {
  const fullDistribution =
    await verifyCppCuteBrowserFullDistributionReproducibilityResource(
      cppCuteBrowserFullDistributionReproducibilityResourceBytes(),
    );
  return {
    producer: await createCurrentPayloadExternallyRootedProducer(
      fullDistribution,
    ),
    fullDistribution,
    convergence:
      await verifyCppCuteBrowserExactDistributionConvergenceResource(
        cppCuteBrowserExactDistributionConvergenceResourceBytes(),
      ),
  };
}
