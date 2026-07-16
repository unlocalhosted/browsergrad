import {
  canonicalCppCuteAotRunnerReceiptBytes,
  verifyCppCuteAotRunnerReceipt,
} from "../../../src/cpp_cute_aot_receipt.js";
import { canonicalCppCuteFrontendArtifactBytes } from "../../../src/cpp_cute_frontend_artifact.js";
import type { PreparedCppCuteFrontendProfile } from "../../../src/cpp_cute_frontend_profile.js";
import type {
  CppCuteAotExecutionEnvironmentLayer,
  PreparedCppCuteAotExecutionEnvironment,
} from "../../../src/cpp_cute_aot_environment.js";
import {
  prepareCppCuteAotOfflineRun,
  type PreparedCppCuteAotOfflineRun,
} from "../../../src/cpp_cute_aot_runner_plan.js";
import {
  artifactCompatibleProfileOptions,
  type CppCuteProfileFixtureOptions,
} from "./cpp_cute_frontend_fixtures.js";
import { createCppCuteAotReceiptFixture } from "./cpp_cute_aot_receipt_fixtures.js";
import { createCppCuteAotExecutionEnvironmentFixture } from "./cpp_cute_aot_environment_fixtures.js";

export interface CppCuteAotRunnerFixture {
  readonly plan: PreparedCppCuteAotOfflineRun;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly executionEnvironment: PreparedCppCuteAotExecutionEnvironment;
  readonly artifactBytes: Uint8Array;
  readonly receiptBytes: Uint8Array;
}

export interface CppCuteAotRunnerFixtureOptions {
  readonly outcome?: "accepted" | "rejected";
  readonly environmentLayers?: readonly CppCuteAotExecutionEnvironmentLayer[];
}

export async function createCppCuteAotRunnerFixture(
  profileOverrides: Partial<CppCuteProfileFixtureOptions> = {},
  options: CppCuteAotRunnerFixtureOptions = {},
): Promise<CppCuteAotRunnerFixture> {
  const profileOptions = { ...artifactCompatibleProfileOptions("d".repeat(64)), ...profileOverrides };
  const environmentLayers = options.environmentLayers;
  const environmentFixture = await createCppCuteAotExecutionEnvironmentFixture({
    profile: profileOptions,
    ...(environmentLayers === undefined ? {} : {
      mutateBody: (body) => {
        (body.image as { layers: readonly CppCuteAotExecutionEnvironmentLayer[] }).layers =
          structuredClone(environmentLayers);
      },
    }),
  });
  const profile = environmentFixture.profile;
  const receiptFixture = await createCppCuteAotReceiptFixture(
    profile,
    environmentFixture.environment,
    options.outcome ?? "accepted",
  );
  const plan = await prepareCppCuteAotOfflineRun(receiptFixture.metadata, environmentFixture.environment);
  const receipt = await verifyCppCuteAotRunnerReceipt(
    receiptFixture.metadata,
    environmentFixture.environment,
    receiptFixture.requestBinding,
    receiptFixture.receipt,
  );
  return {
    plan,
    profile,
    executionEnvironment: environmentFixture.environment,
    artifactBytes: canonicalCppCuteFrontendArtifactBytes(receiptFixture.artifact),
    receiptBytes: canonicalCppCuteAotRunnerReceiptBytes(receipt),
  };
}
