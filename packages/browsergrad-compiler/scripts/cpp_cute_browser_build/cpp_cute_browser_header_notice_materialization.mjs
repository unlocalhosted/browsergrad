import { createHash } from "node:crypto";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  cppCuteBrowserHeaderInputProjectionId,
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  materializeCppCuteBrowserDistributionOutputFiles,
} from "./cpp_cute_browser_distribution_output_files.mjs";
import {
  requireCppCuteBrowserHeaderDistributionReviewInputAuthority,
} from "./cpp_cute_browser_header_distribution_review_input.mjs";
import {
  copyCppCuteBrowserVerifiedHeaderNoticeBytes,
  requireCppCuteBrowserHeaderNoticeVerificationAuthority,
} from "./cpp_cute_browser_header_notice_verification.mjs";
import {
  requireCppCuteBrowserHeaderPackMaterializationAuthority,
} from "./cpp_cute_browser_header_pack_materialization.mjs";

export const CPP_CUTE_BROWSER_HEADER_NOTICE_MATERIALIZATION_SCHEMA =
  "browsergrad.compiler.cpp-cute.browser-header-notice-materialization";
export const CPP_CUTE_BROWSER_AGGREGATE_NOTICE_FORMAT =
  "browsergrad.compiler.cpp-cute.distribution-notices.v2";

const ERROR_CODE = "BG-COMPILER-CPP-CUTE-BROWSER-HEADER-NOTICE-MATERIALIZATION";
const MATERIALIZATION_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-header-notice-materialization.v2";
const AGGREGATE_OUTPUT_PATH = "assets/browsergrad-cpp-cute/THIRD_PARTY_NOTICES.txt";
const TEXT_ENCODER = new TextEncoder();
const SAFE_METADATA = /^[\x20-\x7e]+$/u;
const NOTICE_MATERIALIZATIONS = new WeakSet();

export class CppCuteBrowserHeaderNoticeMaterializationError extends Error {
  constructor(path, message, options) {
    super(`${ERROR_CODE}: ${message}`, options);
    this.name = "CppCuteBrowserHeaderNoticeMaterializationError";
    this.code = ERROR_CODE;
    this.path = path;
  }
}

/**
 * Materializes the current build-lock component-license files and one
 * deterministic aggregate notice from verifier-retained byte snapshots. This
 * is exact private output materialization, never legal or release approval.
 */
export async function materializeCppCuteBrowserHeaderDistributionNotices(input) {
  const object = exactObject(
    input,
    ["distributionReviewInput", "materialization", "notices"],
    "$.input",
  );
  try {
    requireCppCuteBrowserHeaderDistributionReviewInputAuthority(object.distributionReviewInput);
    requireCppCuteBrowserHeaderPackMaterializationAuthority(object.materialization);
    requireCppCuteBrowserHeaderNoticeVerificationAuthority(object.notices);
  } catch (cause) {
    invalid("$.input", "expected one live exact distribution-output authority chain", { cause });
  }
  const buildInputLock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  const body = unwrapPreparedCppCuteBrowserBuildInputLock(buildInputLock).lock.body;
  const headerInputProjectionId =
    await cppCuteBrowserHeaderInputProjectionId(buildInputLock);
  assertIdentityChain(object, buildInputLock, headerInputProjectionId);
  const outputPolicy = exactOutputPolicy(body, object.notices);
  const componentBytes = object.notices.notices.map((notice) => Object.freeze({
    componentId: notice.componentId,
    outputPath: notice.noticeOutputPath,
    bytes: copyCppCuteBrowserVerifiedHeaderNoticeBytes(object.notices, notice.componentId),
  }));
  const aggregateBytes = composeAggregateNoticeBytes(
    headerInputProjectionId,
    object.notices.notices,
    componentBytes,
  );
  const existingOutputs = [
    ...object.materialization.outputs.map((output) => Object.freeze({
      outputPath: output.outputPath,
      sha256: output.packSha256,
      byteLength: output.packByteLength,
    })),
    Object.freeze({
      outputPath: object.distributionReviewInput.outputPath,
      sha256: object.distributionReviewInput.reviewInputSha256,
      byteLength: object.distributionReviewInput.reviewInputByteLength,
    }),
  ];
  let outputMaterialization;
  try {
    outputMaterialization = await materializeCppCuteBrowserDistributionOutputFiles({
      outputRoot: object.materialization.outputRoot,
      existingOutputs,
      outputs: [
        ...componentBytes.map((component) => Object.freeze({
          outputPath: component.outputPath,
          bytes: component.bytes,
        })),
        Object.freeze({ outputPath: outputPolicy.aggregate.path, bytes: aggregateBytes }),
      ],
    });
  } catch (cause) {
    invalid("$.output", "failed to materialize the exact distribution notice set", { cause });
  }
  const outputByPath = new Map(outputMaterialization.outputs.map((output) => [output.outputPath, output]));
  const componentOutputs = Object.freeze(object.notices.notices.map((notice, ordinal) => {
    const output = outputByPath.get(notice.noticeOutputPath);
    if (output === undefined || output.sha256 !== notice.noticeSha256 ||
        output.byteLength !== notice.noticeByteLength) {
      invalid(`$.output.componentOutputs[${ordinal}]`, "component notice output identity differs from policy");
    }
    return Object.freeze({
      ordinal,
      componentId: notice.componentId,
      outputPath: output.outputPath,
      sha256: output.sha256,
      byteLength: output.byteLength,
    });
  }));
  const aggregateOutput = outputByPath.get(outputPolicy.aggregate.path);
  if (aggregateOutput === undefined || aggregateOutput.sha256 !== sha256(aggregateBytes) ||
      aggregateOutput.byteLength !== String(aggregateBytes.byteLength)) {
    invalid("$.output.aggregate", "aggregate notice output identity differs from canonical bytes");
  }
  if (outputByPath.size !== componentOutputs.length + 1) {
    invalid("$.output", "notice output materializer returned an unexpected file set");
  }
  const aggregateReport = Object.freeze({
    outputPath: aggregateOutput.outputPath,
    sha256: aggregateOutput.sha256,
    byteLength: aggregateOutput.byteLength,
    format: CPP_CUTE_BROWSER_AGGREGATE_NOTICE_FORMAT,
    componentCount: componentOutputs.length,
    thirdPartyComponentCount: componentOutputs.filter(
      (output) => output.componentId !== "browsergrad-compiler",
    ).length,
  });
  const materializationHash = sha256(canonicalJsonBytes({
    domain: MATERIALIZATION_HASH_DOMAIN,
    headerInputProjectionId,
    distributionReviewInputId: object.distributionReviewInput.reviewInputId,
    inventoryId: object.materialization.inventoryId,
    outputFileMaterializationId: outputMaterialization.materializationId,
    componentOutputs,
    aggregateOutput: aggregateReport,
  }));
  const report = Object.freeze({
    schema: CPP_CUTE_BROWSER_HEADER_NOTICE_MATERIALIZATION_SCHEMA,
    version: 2,
    noticeMaterializationId:
      `bg.cpp.browser-header-notice-materialization.sha256.${materializationHash}`,
    authority: "exact-private-distribution-notice-materialization-only",
    buildInputLockId: buildInputLock.lockId,
    buildInputLockResourceSha256: buildInputLock.resourceSha256,
    headerInputProjectionId,
    distributionReviewInputId: object.distributionReviewInput.reviewInputId,
    inventoryId: object.materialization.inventoryId,
    outputFileMaterializationId: outputMaterialization.materializationId,
    outputRoot: object.materialization.outputRoot,
    componentOutputs,
    aggregateOutput: aggregateReport,
    totals: Object.freeze({
      componentFileCount: componentOutputs.length,
      componentByteLength: componentOutputs.reduce(
        (total, output) => total + BigInt(output.byteLength),
        0n,
      ).toString(),
      aggregateFileCount: 1,
      aggregateByteLength: aggregateReport.byteLength,
      materializedFileCount: outputMaterialization.totals.materializedFileCount,
      materializedByteLength: outputMaterialization.totals.materializedByteLength,
    }),
    unresolvedNotices: object.notices.unresolvedNotices,
    claims: Object.freeze({
      exactApprovedDistributionNoticeBytesMaterialized: true,
      exactComponentLicenseOutputSetMaterialized: true,
      deterministicAggregateNoticeMaterialized: true,
      exactFinalDistributionOutputTreeVerified: true,
      allHeaderNoticesResolved: false,
      externalDistributedFileLicenseMapReviewed: false,
      licenseReviewComplete: false,
      distributionAuthorized: false,
      releaseReady: false,
    }),
  });
  NOTICE_MATERIALIZATIONS.add(report);
  return report;
}

export function requireCppCuteBrowserHeaderNoticeMaterializationAuthority(report) {
  if (typeof report !== "object" || report === null || !NOTICE_MATERIALIZATIONS.has(report)) {
    invalid("$.report", "expected materializer-issued distribution-notice authority");
  }
}

export function canonicalCppCuteBrowserHeaderNoticeMaterializationBytes(report) {
  requireCppCuteBrowserHeaderNoticeMaterializationAuthority(report);
  return canonicalJsonBytes(report);
}

function assertIdentityChain(object, buildInputLock, headerInputProjectionId) {
  const { distributionReviewInput, materialization, notices } = object;
  if (notices.buildInputLockId !== buildInputLock.lockId ||
      notices.buildInputLockResourceSha256 !== buildInputLock.resourceSha256 ||
      distributionReviewInput.buildInputLockId !== buildInputLock.lockId ||
      distributionReviewInput.buildInputLockResourceSha256 !== buildInputLock.resourceSha256 ||
      distributionReviewInput.headerInputProjectionId !== headerInputProjectionId ||
      materialization.buildInputLockId !== buildInputLock.lockId ||
      materialization.buildInputLockResourceSha256 !== buildInputLock.resourceSha256 ||
      materialization.headerInputProjectionId !== headerInputProjectionId ||
      notices.headerInputProjectionId !== headerInputProjectionId ||
      distributionReviewInput.inventoryId !== materialization.inventoryId) {
    invalid("$.input", "notice authorities do not form one exact current identity chain");
  }
  if (materialization.outputs.length !== 5 || distributionReviewInput.outputPath !==
      "assets/browsergrad-cpp-cute/license-inventory.json") {
    invalid("$.input", "notice materialization requires the exact five-pack review-input closure");
  }
}

function exactOutputPolicy(body, notices) {
  const outputs = body.recipe.distributedOutputPlan.outputs;
  const componentOutputs = outputs.filter((output) => output.role === "component-license");
  const aggregateOutputs = outputs.filter((output) => output.role === "third-party-notices");
  const expectedComponentPaths = notices.notices
    .map((notice) => notice.noticeOutputPath)
    .sort(compareUtf8);
  const actualComponentPaths = componentOutputs.map((output) => output.path).sort(compareUtf8);
  if (!sameStrings(actualComponentPaths, expectedComponentPaths) ||
      componentOutputs.some((output) => output.mediaType !== "text/plain" ||
        output.reproducibilityClass !== "deterministic-subject")) {
    invalid("$.buildInputLock", "component-license output policy differs from the verified notice set");
  }
  const aggregate = aggregateOutputs[0];
  if (aggregateOutputs.length !== 1 || aggregate?.path !== AGGREGATE_OUTPUT_PATH ||
      aggregate.mediaType !== "text/plain" ||
      aggregate.reproducibilityClass !== "deterministic-subject") {
    invalid("$.buildInputLock", "aggregate notice output policy is not exact");
  }
  return Object.freeze({ componentOutputs: Object.freeze(componentOutputs), aggregate });
}

function composeAggregateNoticeBytes(headerInputProjectionId, notices, componentBytes) {
  metadata(headerInputProjectionId, "$.headerInputProjectionId");
  if (notices.length !== componentBytes.length || notices.length === 0) {
    invalid("$.notices", "aggregate notice inputs differ from the verified component set");
  }
  const thirdPartyCount = notices.filter((notice) => notice.componentId !== "browsergrad-compiler").length;
  const chunks = [TEXT_ENCODER.encode(
    `BrowserGrad Compiler Distribution Notices\n` +
    `Format: ${CPP_CUTE_BROWSER_AGGREGATE_NOTICE_FORMAT}\n` +
    `Header-Input-Projection-ID: ${headerInputProjectionId}\n` +
    `Notice-Count: ${notices.length}\n` +
    `Third-Party-Notice-Count: ${thirdPartyCount}\n` +
    `External-File-License-Review: pending\n` +
    `Distribution-Authorization: not-granted\n\n` +
    `This file mechanically aggregates verifier-bound component notice bytes.\n` +
    `It does not complete external file-level review or authorize distribution.\n\n`,
  )];
  for (const [index, notice] of notices.entries()) {
    const retained = componentBytes[index];
    if (retained === undefined || retained.componentId !== notice.componentId ||
        retained.outputPath !== notice.noticeOutputPath ||
        retained.bytes.byteLength !== Number(notice.noticeByteLength) ||
        sha256(retained.bytes) !== notice.noticeSha256) {
      invalid(`$.notices[${index}]`, "aggregate notice bytes differ from verifier authority");
    }
    const values = [
      notice.componentId,
      notice.licenseExpression,
      notice.upstreamSourcePath,
      notice.noticeOutputPath,
      notice.noticeSha256,
      notice.noticeByteLength,
      ...notice.appliesTo,
    ];
    values.forEach((value, valueIndex) => metadata(value, `$.notices[${index}].metadata[${valueIndex}]`));
    const ordinal = String(index).padStart(4, "0");
    chunks.push(TEXT_ENCODER.encode(
      `===== BEGIN NOTICE ${ordinal} =====\n` +
      `Component-ID: ${notice.componentId}\n` +
      `License-Expression: ${notice.licenseExpression}\n` +
      `Upstream-Source-Path: ${notice.upstreamSourcePath}\n` +
      `Component-Notice-Output-Path: ${notice.noticeOutputPath}\n` +
      `Applies-To: ${notice.appliesTo.join(",")}\n` +
      `Notice-SHA256: ${notice.noticeSha256}\n` +
      `Notice-Byte-Length: ${notice.noticeByteLength}\n\n`,
    ));
    chunks.push(retained.bytes);
    if (retained.bytes.at(-1) !== 0x0a) chunks.push(new Uint8Array([0x0a]));
    chunks.push(TEXT_ENCODER.encode(`===== END NOTICE ${ordinal} =====\n\n`));
  }
  const byteLength = chunks.reduce((total, chunk) => total + chunk.byteLength, 0);
  const bytes = new Uint8Array(byteLength);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function exactObject(value, keys, diagnosticPath) {
  if (typeof value !== "object" || value === null || Object.getPrototypeOf(value) !== Object.prototype) {
    invalid(diagnosticPath, "expected one plain data record");
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const actual = Reflect.ownKeys(descriptors);
  if (actual.length !== keys.length ||
      actual.some((key) => typeof key !== "string" || !keys.includes(key))) {
    invalid(diagnosticPath, `expected only ${keys.join(", ")}`);
  }
  const result = {};
  for (const key of keys) {
    const descriptor = descriptors[key];
    if (descriptor === undefined || !("value" in descriptor)) {
      invalid(`${diagnosticPath}.${key}`, "expected data property");
    }
    result[key] = descriptor.value;
  }
  return result;
}

function metadata(value, diagnosticPath) {
  if (typeof value !== "string" || value.length === 0 || !SAFE_METADATA.test(value)) {
    invalid(diagnosticPath, "expected one nonempty printable ASCII metadata value");
  }
}

function sameStrings(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function compareUtf8(left, right) {
  return Buffer.compare(Buffer.from(left, "utf8"), Buffer.from(right, "utf8"));
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function invalid(path, message, options) {
  throw new CppCuteBrowserHeaderNoticeMaterializationError(path, message, options);
}
