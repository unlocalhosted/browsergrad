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
  verifyCppCuteBrowserWorkerBundle,
} from "./cpp_cute_browser_worker_bundle.js";
import {
  CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_RESOURCE,
} from "./resources/cpp_cute_browser_strict_compile_observation_v1.js";

export const CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_SHA256 =
  "bf4d378a92eda260a120da15651deac8d42c7324de490ad1009224a3e7761496";
export const CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_BYTE_LENGTH =
  2_886;
export const CPP_CUTE_BROWSER_STRICT_COMPILE_SOURCE_REVISION =
  "559a0586a172b536179a8c69b8848e5293bf99d4";
export const CPP_CUTE_BROWSER_STRICT_COMPILE_WORKER_BUNDLE_SHA256 =
  "eb7df701054a82f59486c011e9a861e1565525c688e402fc1d3fe1724f2530f6";
const STRICT_OBSERVATION_HEADER_REPRODUCIBILITY_ID =
  "bg.cpp.browser-header-distribution-reproducibility.sha256.43f703672ddbeaf1e6e6d544e3ed50721a2585e947b5d0a1e624293cac80d449";
const STRICT_OBSERVATION_HEADER_OUTPUT_VERIFICATION_ID =
  "bg.cpp.distribution-output-file-verification.sha256.1cc298cf70ed624df258a14b0eb687c6a0666a14cdd4e5d208674f6c0f7fb3df";
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

const BUILTIN_RESOURCE_BYTES = canonicalJsonBytes(
  CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_RESOURCE,
);
const VERIFIED_OBSERVATIONS = new WeakSet<object>();

declare const verifiedCppCuteBrowserStrictCompileObservationBrand:
  unique symbol;

/**
 * Exact package authority for one strict real-browser compilation observation.
 * It proves Worker execution and an accepted source-derived Artifact V3 only.
 * Producer trust, licensing, lowering, backend execution, and release remain
 * separate and false.
 */
export interface VerifiedCppCuteBrowserStrictCompileObservation {
  readonly [verifiedCppCuteBrowserStrictCompileObservationBrand]: true;
  readonly authority: "package-pinned-strict-browser-compile-observation-only";
  readonly resourceSha256:
    typeof CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_SHA256;
  readonly resourceByteLength:
    typeof CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_BYTE_LENGTH;
  readonly sourceRevision:
    typeof CPP_CUTE_BROWSER_STRICT_COMPILE_SOURCE_REVISION;
  readonly workerBundleSha256:
    typeof CPP_CUTE_BROWSER_STRICT_COMPILE_WORKER_BUNDLE_SHA256;
  readonly evidenceId:
    typeof CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_RESOURCE.execution.evidenceId;
  readonly artifactId:
    typeof CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_RESOURCE.execution.artifactId;
  readonly candidateId:
    typeof CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_RESOURCE.semanticCandidate.candidateId;
  readonly layoutSemanticHash:
    typeof CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_RESOURCE.semanticCandidate.layoutSemanticHash;
  readonly reproducibleWasmMatched: true;
  readonly packagePinnedHeaderPacksMatched: true;
  readonly rawWasmVerified: true;
  readonly exactInterfaceConformanceObserved: true;
  readonly strictBrowserCompileObserved: true;
  readonly workerExecutionObserved: true;
  readonly sharedLayoutSemanticsPrepared: true;
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

/** Returns a disposable copy of the exact checked-in strict observation. */
export function cppCuteBrowserStrictCompileObservationResourceBytes():
Uint8Array {
  return new Uint8Array(BUILTIN_RESOURCE_BYTES);
}

/**
 * Admits only the exact package resource and cross-binds its Wasm, headers,
 * Worker bundle, artifact, and retained false authority claims.
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
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-STRICT-COMPILE-EVIDENCE-HASH-MISMATCH",
      "$bytes",
      "evidence bytes differ from the exact package resource",
    );
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
  const resource = CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_V1_RESOURCE;
  const currentWorkerSha256: string = worker.sha256;
  if (currentWorkerSha256 !==
      CPP_CUTE_BROWSER_STRICT_COMPILE_WORKER_BUNDLE_SHA256) {
    mismatch("$.workerBundle", "observation does not bind the current package Worker");
  }
  const observedWasmSha256: string = resource.inputs.wasmSha256;
  if (observedWasmSha256 !== wasm.wasmSha256 ||
      resource.inputs.wasmAuthority !==
        "package-pinned-two-clean-build-output" ||
      !resource.inputs.pinnedReproducibleWasmMatched ||
      resource.inputs.untrustedDiagnosticWasm) {
    mismatch("$.inputs.wasmSha256", "observation does not bind the reproducible package Wasm");
  }
  if (resource.inputs.headerDistributionReproducibilityId !==
        STRICT_OBSERVATION_HEADER_REPRODUCIBILITY_ID ||
      resource.inputs.headerDistributionOutputVerificationId !==
        STRICT_OBSERVATION_HEADER_OUTPUT_VERIFICATION_ID ||
      !sameStrictObservationHeaderPacks(headers.outputs) ||
      !resource.inputs.packagePinnedHeaderPacksMatched ||
      resource.inputs.packCount !== 5 ||
      resource.inputs.installedFileCount !== 5_788) {
    mismatch("$.inputs", "observation does not bind the package header distribution");
  }
  if (resource.schema !==
        "browsergrad.compiler.cpp-cute.browser-real-compile-observation" ||
      resource.version !== 1 ||
      resource.outcome !== "compiled" ||
      resource.authority !==
        "local-real-browser-worker-execution-observation-only" ||
      resource.execution.artifactOutcome !== "accepted" ||
      resource.execution.artifactId !==
        `bg.artifact.cpp-cute-frontend.sha256.${resource.execution.artifactHash}` ||
      resource.execution.acceptedTerminalMessages !== "1" ||
      !resource.execution.rawWasmVerified ||
      !resource.execution.exactInterfaceConformanceObserved ||
      !resource.execution.verifierWorkerExecutionObserved ||
      !resource.execution.workerExecutionObserved ||
      !resource.workerExecutionObserved ||
      !resource.semanticCandidate.sharedLayoutSemanticsPrepared ||
      resource.semanticCandidate.coordinateRank !== 2) {
    mismatch("$.execution", "strict browser execution claims are inconsistent");
  }
  if (resource.headerDistributionLicenseApproved ||
      resource.producerTrusted ||
      resource.loweringAuthorityMinted ||
      resource.backendExecutionAuthorized ||
      resource.releaseReady) {
    mismatch("$", "strict observation widened an independent authority claim");
  }

  const authority = Object.freeze({
    authority: "package-pinned-strict-browser-compile-observation-only",
    resourceSha256:
      CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_SHA256,
    resourceByteLength:
      CPP_CUTE_BROWSER_STRICT_COMPILE_OBSERVATION_RESOURCE_BYTE_LENGTH,
    sourceRevision: CPP_CUTE_BROWSER_STRICT_COMPILE_SOURCE_REVISION,
    workerBundleSha256:
      CPP_CUTE_BROWSER_STRICT_COMPILE_WORKER_BUNDLE_SHA256,
    evidenceId: resource.execution.evidenceId,
    artifactId: resource.execution.artifactId,
    candidateId: resource.semanticCandidate.candidateId,
    layoutSemanticHash: resource.semanticCandidate.layoutSemanticHash,
    reproducibleWasmMatched: true,
    packagePinnedHeaderPacksMatched: true,
    rawWasmVerified: true,
    exactInterfaceConformanceObserved: true,
    strictBrowserCompileObserved: true,
    workerExecutionObserved: true,
    sharedLayoutSemanticsPrepared: true,
    headerDistributionLicenseApproved: false,
    producerTrusted: false,
    loweringAuthorityMinted: false,
    backendExecutionAuthorized: false,
    releaseReady: false,
  }) as VerifiedCppCuteBrowserStrictCompileObservation;
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
  const packs = outputs.filter((output) => output.outputPath.endsWith(".headers.bgvfs"));
  return packs.length === STRICT_OBSERVATION_HEADER_PACKS.length &&
    packs.every((output, index) => {
      const expected = STRICT_OBSERVATION_HEADER_PACKS[index];
      return expected !== undefined &&
        output.outputPath === expected.outputPath &&
        output.sha256 === expected.sha256 &&
        output.byteLength === expected.byteLength;
    });
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
      "SHA-256 is unavailable",
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
