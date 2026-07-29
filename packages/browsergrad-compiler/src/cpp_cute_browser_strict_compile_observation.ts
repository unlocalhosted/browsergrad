import {
  canonicalJsonBytes,
  sha256Hex,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  cppCuteBrowserHeaderDistributionReproducibilityResourceBytes,
  verifyCppCuteBrowserHeaderDistributionReproducibilityResource,
} from "./cpp_cute_browser_header_distribution_reproducibility.js";
import {
  cppCuteBrowserReproducibilityResourceBytes,
  verifyCppCuteBrowserReproducibilityResource,
} from "./cpp_cute_browser_reproducibility.js";
import {
  CPP_CUTE_BROWSER_REAL_COMPILE_CASES,
  CPP_CUTE_BROWSER_REAL_COMPILE_CASE_IDS,
} from "./cpp_cute_browser_real_compile_cases.js";
import {
  verifyCppCuteBrowserWorkerBundle,
} from "./cpp_cute_browser_worker_bundle.js";
import {
  CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_RESOURCE,
} from "./resources/cpp_cute_browser_strict_compile_observation_v1.js";
import {
  CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_BYTE_LENGTH,
  CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_SHA256,
  CPP_CUTE_BROWSER_STRICT_COMPILE_SOURCE_REVISION,
  CPP_CUTE_BROWSER_STRICT_COMPILE_VERIFIER_WORKER_BUNDLE_SHA256,
  CPP_CUTE_BROWSER_STRICT_COMPILE_WORKER_BUNDLE_SHA256,
} from "./resources/cpp_cute_browser_strict_compile_observation_identity_v1.js";

export {
  CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_BYTE_LENGTH,
  CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_SHA256,
  CPP_CUTE_BROWSER_STRICT_COMPILE_SOURCE_REVISION,
  CPP_CUTE_BROWSER_STRICT_COMPILE_VERIFIER_WORKER_BUNDLE_SHA256,
  CPP_CUTE_BROWSER_STRICT_COMPILE_WORKER_BUNDLE_SHA256,
};
const STRICT_OBSERVATION_HEADER_REPRODUCIBILITY_ID =
  "bg.cpp.browser-header-distribution-reproducibility.sha256.4d4c054fd4c93dbdbdef9581eeac52b037af3425e6a1c7eff8acc585abce1e55";
const STRICT_OBSERVATION_HEADER_OUTPUT_VERIFICATION_ID =
  "bg.cpp.distribution-output-file-verification.sha256.5bc2231523c4537b30dac139a40c515b725815eec34d81cd0af79f759b31a441";
const STRICT_OBSERVATION_HEADER_PACKS = Object.freeze([
  Object.freeze({
    outputPath: "assets/browsergrad-cpp-cute/clang-resource.headers.bgvfs",
    sha256: "037acb8aaae9a437ed8275ca608dd92c31a142aa8c882b7ac238e80b3343805e",
    byteLength: "7704705",
  }),
  Object.freeze({
    outputPath: "assets/browsergrad-cpp-cute/cuda-12.6.3.headers.bgvfs",
    sha256: "1917ba19e65d1e3be9dfe23b80c693ba8de5ce8e44538a7d16715cd61ece2cbd",
    byteLength: "18954596",
  }),
  Object.freeze({
    outputPath: "assets/browsergrad-cpp-cute/cutlass-3.7.0.headers.bgvfs",
    sha256: "4f1c39b73f2fa7252628a253f7bb5b1411bdfdada872c5ff733b1b9008d89555",
    byteLength: "21403975",
  }),
  Object.freeze({
    outputPath: "assets/browsergrad-cpp-cute/libcxx-22.1.8.headers.bgvfs",
    sha256: "1f2c5a1e86b04c29b6af33cc3fba0487b9bdcb87affaa44fceb32bec424e7dba",
    byteLength: "12689654",
  }),
  Object.freeze({
    outputPath: "assets/browsergrad-cpp-cute/linux-sysroot.headers.bgvfs",
    sha256: "d04a460dc605703b8e8a104cc5c043e6a7020ca7201991470c615273d43e7ae4",
    byteLength: "8927070",
  }),
]);
const EXPECTED_CASES = Object.freeze(
  CPP_CUTE_BROWSER_REAL_COMPILE_CASE_IDS.map(
    (caseId) => CPP_CUTE_BROWSER_REAL_COMPILE_CASES[caseId],
  ),
);

const BUILTIN_RESOURCE_BYTES = canonicalJsonBytes(
  CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_RESOURCE,
);
const VERIFIED_OBSERVATIONS = new WeakSet<object>();

declare const verifiedCppCuteBrowserStrictCompileObservationBrand:
  unique symbol;

/**
 * One exact case admitted from the package-pinned strict browser compilation
 * matrix. The identities remain observation-only and cannot authorize
 * lowering or execution.
 */
export interface VerifiedCppCuteBrowserStrictCompileCase {
  readonly caseId: (typeof EXPECTED_CASES)[number]["caseId"];
  readonly dtype: (typeof EXPECTED_CASES)[number]["dtype"];
  readonly coordinateRank: number;
  readonly sourceSpanElements: string;
  readonly destinationSpanElements: string;
  readonly evidenceId: string;
  readonly artifactId: string;
  readonly candidateId: string;
}

/**
 * Exact package authority for the strict nine-case layout and dtype browser
 * compilation matrix. It proves source execution and semantic-candidate
 * preparation only. Producer trust, licensing, lowering, backend execution,
 * and release stay false.
 */
export interface VerifiedCppCuteBrowserStrictCompileObservation {
  readonly [verifiedCppCuteBrowserStrictCompileObservationBrand]: true;
  readonly authority:
    "package-pinned-strict-browser-compile-matrix-observation-only";
  readonly resourceSha256:
    typeof CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_SHA256;
  readonly resourceByteLength:
    typeof CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_BYTE_LENGTH;
  readonly sourceRevision:
    typeof CPP_CUTE_BROWSER_STRICT_COMPILE_SOURCE_REVISION;
  readonly workerBundleSha256:
    typeof CPP_CUTE_BROWSER_STRICT_COMPILE_WORKER_BUNDLE_SHA256;
  readonly verifierWorkerBundleSha256:
    typeof CPP_CUTE_BROWSER_STRICT_COMPILE_VERIFIER_WORKER_BUNDLE_SHA256;
  readonly caseCount: 9;
  readonly cases: readonly VerifiedCppCuteBrowserStrictCompileCase[];
  readonly unchangedCpp17CuteRank1Compiled: true;
  readonly unchangedCpp17CuteRank2Compiled: true;
  readonly unchangedCpp17CuteRank3Compiled: true;
  readonly unchangedCpp17CuteRank4Compiled: true;
  readonly unchangedCpp17CuteStridedSliceCompiled: true;
  readonly unchangedCpp17CuteBroadcastCompiled: true;
  readonly unchangedCpp17CuteI32Rank2Compiled: true;
  readonly unchangedCpp17CuteU32BroadcastCompiled: true;
  readonly unchangedCpp17CuteSignedRank2Compiled: true;
  readonly canonicalGate2LayoutFixturesMatched: true;
  readonly reproducibleWasmMatched: true;
  readonly packagePinnedHeaderPacksMatched: true;
  readonly rawWasmVerified: true;
  readonly exactInterfaceConformanceObserved: true;
  readonly workerExecutionObserved: true;
  readonly sharedViewCopySemanticsPrepared: true;
  readonly headerDistributionLicenseApproved: false;
  readonly producerTrusted: false;
  readonly loweringAuthorityMinted: false;
  readonly backendExecutionAuthorized: false;
  readonly releaseReady: false;
}

export type CppCuteBrowserStrictCompileObservationErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-STRICT-COMPILE-EVIDENCE-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-STRICT-COMPILE-EVIDENCE-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-STRICT-COMPILE-EVIDENCE-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-STRICT-COMPILE-EVIDENCE-UNVERIFIED";

export class CppCuteBrowserStrictCompileObservationError extends Error {
  constructor(
    readonly code: CppCuteBrowserStrictCompileObservationErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserStrictCompileObservationError";
  }
}

/** Returns a disposable copy of the exact checked-in strict matrix. */
export function cppCuteBrowserStrictCompileObservationResourceBytes():
Uint8Array {
  return new Uint8Array(BUILTIN_RESOURCE_BYTES);
}

/**
 * Admits only the exact package resource and independently cross-binds its
 * extractor, complete header-pack set, Worker module, four artifacts, and
 * retained false authority claims.
 */
export async function verifyCppCuteBrowserStrictCompileObservationResource(
  value: Uint8Array,
): Promise<VerifiedCppCuteBrowserStrictCompileObservation> {
  let inspected: ReturnType<typeof inspectUnsharedPlainUint8Array>;
  try {
    inspected = inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-STRICT-COMPILE-EVIDENCE-INVALID",
      "$bytes",
      "evidence must be one plain unshared Uint8Array",
      { cause },
    );
  }
  if (inspected.byteLength !==
      CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_BYTE_LENGTH) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-STRICT-COMPILE-EVIDENCE-RESOURCE-LIMIT",
      "$bytes.byteLength",
      "evidence byte length differs from the exact package resource",
    );
  }
  const snapshot = copyInspectedUnsharedUint8Array(value, inspected);
  if (!equalBytes(snapshot, BUILTIN_RESOURCE_BYTES)) {
    mismatch("$bytes", "evidence bytes differ from the exact package resource");
  }
  const resourceSha256 = await hash(snapshot, "$bytes.sha256");
  if (resourceSha256 !==
      CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_SHA256) {
    mismatch("$bytes.sha256", "evidence digest differs from its package identity");
  }

  const [wasm, headers, worker] = await Promise.all([
    verifyCppCuteBrowserReproducibilityResource(
      cppCuteBrowserReproducibilityResourceBytes(),
    ),
    verifyCppCuteBrowserHeaderDistributionReproducibilityResource(
      cppCuteBrowserHeaderDistributionReproducibilityResourceBytes(),
    ),
    verifyCppCuteBrowserWorkerBundle(),
  ]);
  if (String(worker.sha256) !==
      String(CPP_CUTE_BROWSER_STRICT_COMPILE_WORKER_BUNDLE_SHA256)) {
    mismatch("$.workerBundle", "observation does not bind the current package Worker");
  }
  if (headers.reproducibilityId !==
        STRICT_OBSERVATION_HEADER_REPRODUCIBILITY_ID ||
      headers.outputVerificationId !==
        STRICT_OBSERVATION_HEADER_OUTPUT_VERIFICATION_ID ||
      !sameStrictObservationHeaderPacks(headers.outputs)) {
    mismatch("$.headers", "observation does not bind the package header distribution");
  }

  const resource = CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_RESOURCE;
  if (resource.schema !==
        "browsergrad.compiler.cpp-cute.browser-real-compile-matrix-observation" ||
      resource.version !== 2 ||
      resource.authority !==
        "local-real-browser-worker-matrix-observation-only" ||
      resource.caseCount !== EXPECTED_CASES.length ||
      resource.cases.length !== EXPECTED_CASES.length) {
    mismatch("$", "strict matrix envelope is inconsistent");
  }
  if (resource.packageBinding.compilerWorkerSha256 !== worker.sha256 ||
      resource.packageBinding.compilerWorkerSha256 !==
        CPP_CUTE_BROWSER_STRICT_COMPILE_WORKER_BUNDLE_SHA256 ||
      resource.packageBinding.verifierWorkerSha256 !==
        CPP_CUTE_BROWSER_STRICT_COMPILE_VERIFIER_WORKER_BUNDLE_SHA256 ||
      resource.packageBinding.matrixSourceRevision !==
        CPP_CUTE_BROWSER_STRICT_COMPILE_SOURCE_REVISION ||
      resource.packageBinding.workerBundleAuthority !==
        "package-owned-zero-import-module-bytes") {
    mismatch(
      "$.packageBinding",
      "strict matrix does not bind the current package Worker pair and source",
    );
  }
  const headerPackByteLength = strictObservationHeaderPackByteLength();
  const evidenceIds = new Set<string>();
  const artifactIds = new Set<string>();
  const candidateIds = new Set<string>();
  const assetSetIds = new Set<string>();
  for (const [index, expected] of EXPECTED_CASES.entries()) {
    const observed = resource.cases[index];
    if (observed === undefined ||
        observed.schema !==
          "browsergrad.compiler.cpp-cute.browser-real-compile-observation" ||
        observed.version !== 2 ||
        observed.outcome !== "compiled" ||
        observed.authority !==
          "local-real-browser-worker-execution-observation-only" ||
        observed.source.caseId !== expected.caseId ||
        observed.source.virtualPath !== expected.virtualPath ||
        observed.source.sourceSha256 !== expected.sourceSha256 ||
        observed.source.syntax !== "unchanged-cpp17-cute" ||
        observed.source.selectedDeclaration !== "copy_views") {
      mismatch(`$.cases[${index}].source`, "strict source identity is inconsistent");
    }
    if (String(observed.inputs.wasmSha256) !== wasm.wasmSha256 ||
        observed.inputs.wasmAuthority !==
          "package-pinned-two-clean-build-output" ||
        !observed.inputs.pinnedReproducibleWasmMatched ||
        observed.inputs.untrustedDiagnosticWasm ||
        observed.inputs.headerDistributionReproducibilityId !==
          headers.reproducibilityId ||
        observed.inputs.headerDistributionOutputVerificationId !==
          headers.outputVerificationId ||
        !observed.inputs.packagePinnedHeaderPacksMatched ||
        observed.inputs.externalAssetCount !== 6 ||
        observed.inputs.packCount !== 5 ||
        observed.inputs.installedFileCount !== 5_788 ||
        observed.inputs.totalExternalByteLength !==
          wasm.wasmByteLength + headerPackByteLength) {
      mismatch(`$.cases[${index}].inputs`, "strict input closure is inconsistent");
    }
    if (observed.execution.artifactOutcome !== "accepted" ||
        observed.execution.artifactId !==
          `bg.artifact.cpp-cute-frontend.sha256.${observed.execution.artifactHash}` ||
        observed.execution.acceptedTerminalMessages !== "1" ||
        !observed.execution.rawWasmVerified ||
        !observed.execution.exactInterfaceConformanceObserved ||
        !observed.execution.verifierWorkerExecutionObserved ||
        !observed.execution.workerExecutionObserved ||
        observed.execution.openedSourceFiles !== "1" ||
        observed.execution.compileElapsedMilliseconds <= 0 ||
        observed.execution.totalElapsedMilliseconds <
          observed.execution.compileElapsedMilliseconds) {
      mismatch(`$.cases[${index}].execution`, "strict execution is inconsistent");
    }
    if (observed.semanticCandidate.sourceCoordinateRank !==
          expected.coordinateRank ||
        observed.semanticCandidate.destinationCoordinateRank !==
          expected.coordinateRank ||
        observed.semanticCandidate.sourceSpanElements !==
          expected.sourceSpanElements.toString(10) ||
        observed.semanticCandidate.destinationSpanElements !==
          expected.destinationSpanElements.toString(10) ||
        observed.semanticCandidate.dtype !== expected.dtype ||
        !observed.semanticCandidate.sharedViewCopySemanticsPrepared) {
      mismatch(
        `$.cases[${index}].semanticCandidate`,
        "strict semantic candidate is inconsistent",
      );
    }
    if (observed.headerDistributionLicenseApproved ||
        observed.producerTrusted ||
        observed.loweringAuthorityMinted ||
        observed.backendExecutionAuthorized ||
        observed.releaseReady ||
        !observed.workerExecutionObserved) {
      mismatch(`$.cases[${index}]`, "strict case widened an authority claim");
    }
    evidenceIds.add(observed.execution.evidenceId);
    artifactIds.add(observed.execution.artifactId);
    candidateIds.add(observed.semanticCandidate.candidateId);
    assetSetIds.add(observed.inputs.assetSetSha256);
  }
  if (evidenceIds.size !== EXPECTED_CASES.length ||
      artifactIds.size !== EXPECTED_CASES.length ||
      candidateIds.size !== EXPECTED_CASES.length ||
      assetSetIds.size !== 1) {
    mismatch("$.cases", "strict matrix identities are reused or divergent");
  }
  const claims = resource.claims as typeof resource.claims & {
    readonly unchangedCpp17CuteSignedRank2Compiled?: boolean;
  };
  if (!claims.unchangedCpp17CuteRank1Compiled ||
      !claims.unchangedCpp17CuteRank2Compiled ||
      !claims.unchangedCpp17CuteRank3Compiled ||
      !claims.unchangedCpp17CuteRank4Compiled ||
      !claims.unchangedCpp17CuteStridedSliceCompiled ||
      !claims.unchangedCpp17CuteBroadcastCompiled ||
      !claims.unchangedCpp17CuteI32Rank2Compiled ||
      !claims.unchangedCpp17CuteU32BroadcastCompiled ||
      claims.unchangedCpp17CuteSignedRank2Compiled !== true ||
      !claims.canonicalGate2LayoutFixturesMatched ||
      !claims.packagePinnedHeaderPacksMatched ||
      !claims.pinnedReproducibleWasmMatched ||
      claims.untrustedDiagnosticWasm ||
      !claims.workerExecutionObserved ||
      claims.headerDistributionLicenseApproved ||
      claims.producerTrusted ||
      claims.loweringAuthorityMinted ||
      claims.backendExecutionAuthorized ||
      claims.releaseReady) {
    mismatch("$.claims", "strict matrix claims are inconsistent");
  }

  const cases = Object.freeze(resource.cases.map((observed) => Object.freeze({
    caseId: observed.source.caseId,
    dtype: observed.semanticCandidate.dtype,
    coordinateRank: observed.semanticCandidate.sourceCoordinateRank,
    sourceSpanElements: observed.semanticCandidate.sourceSpanElements,
    destinationSpanElements:
      observed.semanticCandidate.destinationSpanElements,
    evidenceId: observed.execution.evidenceId,
    artifactId: observed.execution.artifactId,
    candidateId: observed.semanticCandidate.candidateId,
  })));
  const authority = Object.freeze({
    authority:
      "package-pinned-strict-browser-compile-matrix-observation-only",
    resourceSha256:
      CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_SHA256,
    resourceByteLength:
      CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_BYTE_LENGTH,
    sourceRevision: CPP_CUTE_BROWSER_STRICT_COMPILE_SOURCE_REVISION,
    workerBundleSha256:
      CPP_CUTE_BROWSER_STRICT_COMPILE_WORKER_BUNDLE_SHA256,
    verifierWorkerBundleSha256:
      CPP_CUTE_BROWSER_STRICT_COMPILE_VERIFIER_WORKER_BUNDLE_SHA256,
    caseCount: 9,
    cases,
    unchangedCpp17CuteRank1Compiled: true,
    unchangedCpp17CuteRank2Compiled: true,
    unchangedCpp17CuteRank3Compiled: true,
    unchangedCpp17CuteRank4Compiled: true,
    unchangedCpp17CuteStridedSliceCompiled: true,
    unchangedCpp17CuteBroadcastCompiled: true,
    unchangedCpp17CuteI32Rank2Compiled: true,
    unchangedCpp17CuteU32BroadcastCompiled: true,
    unchangedCpp17CuteSignedRank2Compiled: true,
    canonicalGate2LayoutFixturesMatched: true,
    reproducibleWasmMatched: true,
    packagePinnedHeaderPacksMatched: true,
    rawWasmVerified: true,
    exactInterfaceConformanceObserved: true,
    workerExecutionObserved: true,
    sharedViewCopySemanticsPrepared: true,
    headerDistributionLicenseApproved: false,
    producerTrusted: false,
    loweringAuthorityMinted: false,
    backendExecutionAuthorized: false,
    releaseReady: false,
  }) as unknown as VerifiedCppCuteBrowserStrictCompileObservation;
  VERIFIED_OBSERVATIONS.add(authority);
  return authority;
}

function sameStrictObservationHeaderPacks(
  outputs: readonly Readonly<{
    outputPath: string;
    sha256: string;
    byteLength: string;
  }>[],
): boolean {
  const packs = outputs.filter((output) =>
    output.outputPath.endsWith(".headers.bgvfs"));
  return packs.length === STRICT_OBSERVATION_HEADER_PACKS.length &&
    packs.every((output, index) => {
      const expected = STRICT_OBSERVATION_HEADER_PACKS[index];
      return expected !== undefined &&
        output.outputPath === expected.outputPath &&
        output.sha256 === expected.sha256 &&
        output.byteLength === expected.byteLength;
    });
}

function strictObservationHeaderPackByteLength(): number {
  let total = 0;
  for (const pack of STRICT_OBSERVATION_HEADER_PACKS) {
    const byteLength = Number(pack.byteLength);
    if (!Number.isSafeInteger(byteLength) || byteLength <= 0) {
      mismatch("$.headers", "package header byte length is invalid");
    }
    total += byteLength;
  }
  return total;
}

export function requireVerifiedCppCuteBrowserStrictCompileObservation(
  authority: VerifiedCppCuteBrowserStrictCompileObservation,
): void {
  if (typeof authority !== "object" || authority === null ||
      !VERIFIED_OBSERVATIONS.has(authority)) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-STRICT-COMPILE-EVIDENCE-UNVERIFIED",
      "$authority",
      "expected verifier-issued strict browser compile authority",
    );
  }
}

async function hash(bytes: Uint8Array, path: string): Promise<string> {
  try {
    return await sha256Hex(bytes);
  } catch (cause) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-STRICT-COMPILE-EVIDENCE-UNVERIFIED",
      path,
      "SHA-256 is unavailable for strict browser compile evidence",
      { cause },
    );
  }
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}

function mismatch(path: string, message: string): never {
  fail(
    "BG-COMPILER-CPP-CUTE-BROWSER-STRICT-COMPILE-EVIDENCE-HASH-MISMATCH",
    path,
    message,
  );
}

function fail(
  code: CppCuteBrowserStrictCompileObservationErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserStrictCompileObservationError(
    code,
    path,
    message,
    options,
  );
}
