import { sha256Hex } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  canonicalCppCuteAotRunnerReceiptBytes,
  verifyCppCuteAotRunnerReceipt,
} from "../../../src/cpp_cute_aot_receipt.js";
import {
  canonicalCppCuteFrontendArtifactBytes,
  deriveCppCuteFrontendArtifactId,
  deriveCppCuteStableId,
  verifyCppCuteFrontendArtifact,
} from "../../../src/cpp_cute_frontend_artifact.js";
import {
  type PreparedCppCuteFrontendProfile,
} from "../../../src/cpp_cute_frontend_profile.js";
import type {
  CppCuteAotExecutionEnvironmentLayer,
  PreparedCppCuteAotExecutionEnvironment,
} from "../../../src/cpp_cute_aot_environment.js";
import {
  computeCppCuteInputHashes,
} from "../../../src/cpp_cute_frontend_verify.js";
import type {
  CppCuteFrontendArtifactV2,
  CppCuteFrontendPayloadV2,
} from "../../../src/cpp_cute_frontend_types.js";
import {
  prepareCppCuteAotOfflineRun,
  type CppCuteAotSourceBlob,
  type PreparedCppCuteAotOfflineRun,
} from "../../../src/cpp_cute_aot_runner_plan.js";
import {
  artifactCompatibleProfileOptions,
  createCppCuteArtifactInput,
  type CppCuteProfileFixtureOptions,
} from "./cpp_cute_frontend_fixtures.js";
import {
  createCppCuteAotReceiptFixture,
} from "./cpp_cute_aot_receipt_fixtures.js";
import { createCppCuteAotExecutionEnvironmentFixture } from "./cpp_cute_aot_environment_fixtures.js";

export interface CppCuteAotRunnerFixture {
  readonly plan: PreparedCppCuteAotOfflineRun;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly executionEnvironment: PreparedCppCuteAotExecutionEnvironment;
  readonly sourceBlob: CppCuteAotSourceBlob;
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
  const profileOptions = {
    ...artifactCompatibleProfileOptions("d".repeat(64)),
    ...profileOverrides,
  };
  const environmentLayers = options.environmentLayers;
  const environmentFixture = await createCppCuteAotExecutionEnvironmentFixture({
    profile: profileOptions,
    ...(environmentLayers === undefined
      ? {}
      : {
          mutateBody: (body) => {
            (body.image as { layers: readonly CppCuteAotExecutionEnvironmentLayer[] }).layers =
              structuredClone(environmentLayers);
          },
        }),
  });
  const profile = environmentFixture.profile;
  const artifactInput = await createRealSourceBackedArtifact(
    profile.compilationContractHash,
    options.outcome ?? "accepted",
  );
  const artifact = await verifyCppCuteFrontendArtifact(artifactInput);
  const receiptFixture = await createCppCuteAotReceiptFixture(
    profile,
    environmentFixture.environment,
    artifact,
  );
  const sourceFile = receiptFixture.receipt.openedInputs.files.find((file) => file.role === "main-source");
  if (sourceFile === undefined) throw new Error("runner fixture lost main source");
  const sourceBytes = fixtureSourceBytes();
  const sourceBlob = Object.freeze({ fileId: sourceFile.fileId, bytes: sourceBytes });
  const plan = await prepareCppCuteAotOfflineRun(
    receiptFixture.job,
    environmentFixture.environment,
    [sourceBlob],
  );
  const receipt = await verifyCppCuteAotRunnerReceipt(
    receiptFixture.job,
    environmentFixture.environment,
    receiptFixture.artifactResource,
    receiptFixture.receipt,
  );
  return {
    plan,
    profile,
    executionEnvironment: environmentFixture.environment,
    sourceBlob,
    artifactBytes: canonicalCppCuteFrontendArtifactBytes(receiptFixture.artifact),
    receiptBytes: canonicalCppCuteAotRunnerReceiptBytes(receipt),
  };
}

async function createRealSourceBackedArtifact(
  compilationContractHash: string,
  outcome: "accepted" | "rejected",
): Promise<CppCuteFrontendArtifactV2> {
  const artifact = await createCppCuteArtifactInput(compilationContractHash);
  const payload = structuredClone(artifact.payload) as CppCuteFrontendPayloadV2;
  const main = payload.inputs.files.find((file) => file.role === "main-source");
  if (main === undefined) throw new Error("runner fixture lost source file");
  const oldFileId = main.fileId;
  const bytes = fixtureSourceBytes();
  const contentSha256 = await sha256Hex(bytes);
  const newFileId = await deriveCppCuteStableId("file", {
    role: main.role,
    virtualPath: main.virtualPath,
    contentSha256,
    byteLength: String(bytes.byteLength),
    owner: main.owner,
    includeRootId: main.includeRootId,
  });
  replaceString(payload, oldFileId, newFileId);
  const rewrittenMain = payload.inputs.files.find((file) => file.fileId === newFileId);
  if (rewrittenMain === undefined) throw new Error("runner fixture failed to rewrite source identity");
  (rewrittenMain as { contentSha256: string }).contentSha256 = contentSha256;
  (rewrittenMain as { byteLength: string }).byteLength = String(bytes.byteLength);
  const hashes = await computeCppCuteInputHashes(payload);
  (payload.inputs as { sourceSetSha256: string }).sourceSetSha256 = hashes.sourceSetSha256;
  (payload.inputs as { headerSetSha256: string }).headerSetSha256 = hashes.headerSetSha256;
  (payload.inputs as { closureSha256: string }).closureSha256 = hashes.closureSha256;
  (payload.extraction as { inputClosureSha256: string }).inputClosureSha256 = hashes.closureSha256;
  if (outcome === "rejected") rejectPayload(payload);
  return {
    ...artifact,
    artifactId: await deriveCppCuteFrontendArtifactId(payload),
    payload,
  };
}

function rejectPayload(payload: CppCuteFrontendPayloadV2): void {
  const diagnostic = payload.diagnostics[0];
  if (diagnostic === undefined) throw new Error("runner fixture lost diagnostic");
  const blockingDiagnosticId = `bg.cpp.diagnostic.sha256.${"1".repeat(64)}`;
  (payload.diagnostics as unknown as Array<unknown>).push({
    diagnosticId: blockingDiagnosticId,
    phase: "artifact-extraction",
    severity: "error",
    code: "browsergrad.cpp-cute:fixture-rejected",
    renderedMessage: "Fixture rejection for offline-runner coverage.",
    location: structuredClone(diagnostic.location),
    subject: structuredClone(diagnostic.subject),
    parentDiagnosticId: null,
  });
  (payload.diagnostics as unknown as Array<{ diagnosticId: string }>).sort((left, right) =>
    left.diagnosticId.localeCompare(right.diagnosticId));
  (payload as { outcome: unknown }).outcome = {
    kind: "rejected",
    blockingDiagnosticIds: [blockingDiagnosticId],
  };
}

function fixtureSourceBytes(): Uint8Array {
  const bytes = new Uint8Array(100);
  for (let index = 1; index < bytes.length; index += 1) bytes[index] = (index * 31) & 0xff;
  return bytes;
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
