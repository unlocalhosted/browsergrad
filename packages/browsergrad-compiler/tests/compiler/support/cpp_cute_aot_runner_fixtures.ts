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
  prepareCppCuteFrontendProfile,
} from "../../../src/cpp_cute_frontend_profile.js";
import {
  computeCppCuteInputHashes,
} from "../../../src/cpp_cute_frontend_verify.js";
import type {
  CppCuteFrontendArtifactV1,
  CppCuteFrontendPayloadV1,
} from "../../../src/cpp_cute_frontend_types.js";
import {
  prepareCppCuteAotOfflineRun,
  type CppCuteAotSourceBlob,
  type PreparedCppCuteAotOfflineRun,
} from "../../../src/cpp_cute_aot_runner_plan.js";
import {
  artifactCompatibleProfileOptions,
  createCppCuteArtifactInput,
  createCppCuteProfileInput,
  type CppCuteProfileFixtureOptions,
} from "./cpp_cute_frontend_fixtures.js";
import {
  createCppCuteAotReceiptFixture,
} from "./cpp_cute_aot_receipt_fixtures.js";

export interface CppCuteAotRunnerFixture {
  readonly plan: PreparedCppCuteAotOfflineRun;
  readonly sourceBlob: CppCuteAotSourceBlob;
  readonly artifactBytes: Uint8Array;
  readonly receiptBytes: Uint8Array;
}

export interface CppCuteAotRunnerFixtureOptions {
  readonly outcome?: "accepted" | "rejected";
}

export async function createCppCuteAotRunnerFixture(
  profileOverrides: Partial<CppCuteProfileFixtureOptions> = {},
  options: CppCuteAotRunnerFixtureOptions = {},
): Promise<CppCuteAotRunnerFixture> {
  const preliminary = await verifyCppCuteFrontendArtifact(await createCppCuteArtifactInput());
  const profileOptions = {
    ...artifactCompatibleProfileOptions(preliminary.headerSetSha256, "d".repeat(64)),
    ...profileOverrides,
  };
  const profile = await prepareCppCuteFrontendProfile(createCppCuteProfileInput(profileOptions));
  const artifactInput = await createRealSourceBackedArtifact(
    profile.profileHash,
    options.outcome ?? "accepted",
  );
  const artifact = await verifyCppCuteFrontendArtifact(artifactInput);
  const receiptFixture = await createCppCuteAotReceiptFixture(profile, artifact);
  const sourceFile = receiptFixture.receipt.openedInputs.files.find((file) => file.role === "main-source");
  if (sourceFile === undefined) throw new Error("runner fixture lost main source");
  const sourceBytes = fixtureSourceBytes();
  const sourceBlob = Object.freeze({ fileId: sourceFile.fileId, bytes: sourceBytes });
  const plan = await prepareCppCuteAotOfflineRun(receiptFixture.job, [sourceBlob]);
  const receipt = await verifyCppCuteAotRunnerReceipt(
    receiptFixture.job,
    receiptFixture.artifactResource,
    receiptFixture.receipt,
  );
  return {
    plan,
    sourceBlob,
    artifactBytes: canonicalCppCuteFrontendArtifactBytes(artifact),
    receiptBytes: canonicalCppCuteAotRunnerReceiptBytes(receipt),
  };
}

async function createRealSourceBackedArtifact(
  profileHash: string,
  outcome: "accepted" | "rejected",
): Promise<CppCuteFrontendArtifactV1> {
  const artifact = await createCppCuteArtifactInput(profileHash);
  const payload = structuredClone(artifact.payload) as CppCuteFrontendPayloadV1;
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
    profileDependency: main.profileDependency,
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

function rejectPayload(payload: CppCuteFrontendPayloadV1): void {
  const diagnostic = payload.diagnostics[0];
  if (diagnostic === undefined) throw new Error("runner fixture lost diagnostic");
  const blockingDiagnosticId = `bg.cpp.diagnostic.sha256.${"1".repeat(64)}`;
  (payload.diagnostics as unknown as Array<unknown>).push({
    diagnosticId: blockingDiagnosticId,
    phase: "artifact-extraction",
    severity: "error",
    code: "browsergrad.cpp-cute:fixture-rejected",
    renderedMessage: "Fixture rejection for offline-runner coverage.",
    primarySpanId: diagnostic.primarySpanId,
    subject: structuredClone(diagnostic.subject),
    parentDiagnosticId: null,
    related: [],
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
