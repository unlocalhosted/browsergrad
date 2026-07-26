import {
  canonicalJsonBytes,
  sha256Hex,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "./cpp_cute_browser_build_lock.js";
import {
  cppCuteBrowserHeaderDistributionReproducibilityResourceBytes,
  verifyCppCuteBrowserHeaderDistributionReproducibilityResource,
} from "./cpp_cute_browser_header_distribution_reproducibility.js";
import {
  cppCuteBrowserReproducibilityResourceBytes,
  verifyCppCuteBrowserReproducibilityResource,
} from "./cpp_cute_browser_reproducibility.js";
import {
  cppCuteBrowserRuntimeAbiManifestResourceBytes,
  decodeCppCuteBrowserRuntimeAbiManifest,
} from "./cpp_cute_browser_runtime_abi.js";
import {
  inspectVerifiedCppCuteBrowserWorkerBundle,
  verifyCppCuteBrowserWorkerBundle,
} from "./cpp_cute_browser_worker_bundle.js";
import {
  cppCuteDiagnosticNormalizationResourceBytes,
  decodeCppCuteDiagnosticNormalization,
} from "./cpp_cute_diagnostic_normalization.js";
import {
  cppCuteSemanticAdapterManifestResourceBytes,
  decodeCppCuteSemanticAdapterManifest,
} from "./cpp_cute_semantic_adapter_manifest.js";
import {
  CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE,
  type CppCuteBrowserFullDistributionReproducibilityOutputV1,
} from
  "./resources/cpp_cute_browser_full_distribution_reproducibility_v1.js";

export const
  CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_SHA256 =
    "4637cf9624ad00dd833b72b832f2c9e25ee0a8ffcdf62d8b95b732791d36a65a";
export const
  CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH =
    8_951;
export const
  CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_OBSERVATION_SOURCE_REVISION =
    "0840801a138df057da236e84099cc12182b40e2c";

const REPRODUCIBILITY_HASH_DOMAIN =
  "browsergrad.compiler.cpp-cute.browser-full-distribution-reproducibility.v1";
const BUILD_SUBJECT_ID =
  /^bg\.cpp\.browser-build-subject\.sha256\.([0-9a-f]{64})$/u;
const ASSET_MANIFEST_ID =
  /^bg\.cpp\.browser-assets\.sha256\.[0-9a-f]{64}$/u;
const METADATA_ID =
  /^bg\.cpp\.browser-distribution-metadata\.sha256\.[0-9a-f]{64}$/u;
const SHA256 = /^[0-9a-f]{64}$/u;
const WIRE_U64 = /^(?:0|[1-9][0-9]*)$/u;
const BUILTIN_RESOURCE_BYTES = canonicalJsonBytes(
  CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE,
);
const VERIFIED_RESOURCES = new WeakSet<object>();

declare const verifiedCppCuteBrowserFullDistributionReproducibilityBrand:
  unique symbol;

/**
 * Package-pinned engineering evidence for two exact complete distribution
 * roots. It proves output-set reproducibility only. A local engineering
 * signature cannot grant externally rooted producer, legal, distribution,
 * execution, backend, or release authority.
 */
export interface VerifiedCppCuteBrowserFullDistributionReproducibility {
  readonly [verifiedCppCuteBrowserFullDistributionReproducibilityBrand]: true;
  readonly authority:
    "package-pinned-full-distribution-reproducibility-only";
  readonly resourceSha256:
    typeof CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_SHA256;
  readonly resourceByteLength:
    typeof CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH;
  readonly observationVerifierSourceRevision:
    typeof CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_OBSERVATION_SOURCE_REVISION;
  readonly materializerSourceRevision:
    typeof CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_OBSERVATION_SOURCE_REVISION;
  readonly producerPolicyScope:
    "local-engineering-reproducibility-only";
  readonly buildInputLockId: string;
  readonly buildInputLockResourceSha256: string;
  readonly reproducibilityId: string;
  readonly deterministicMetadata:
    typeof CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE.deterministicMetadata;
  readonly deterministicOutputs:
    readonly CppCuteBrowserFullDistributionReproducibilityOutputV1[];
  readonly detachedEvidence:
    typeof CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE.detachedEvidence;
  readonly outputCount: 25;
  readonly deterministicSubjectCount: 24;
  readonly detachedEvidenceCount: 1;
  readonly firstByteLength: "103637461";
  readonly secondByteLength: "103637461";
  readonly twoDistinctPrivateOutputRootsVerified: true;
  readonly exactBuildLockOutputPlanMatched: true;
  readonly exactOutputsRehashedInBothRoots: true;
  readonly deterministicSubjectsByteIdentical: true;
  readonly detachedEvidenceBuildSubjectMatched: true;
  readonly fullDistributedOutputSetReproducible: true;
  readonly detachedSignatureVerified: false;
  readonly externallyRootedProducerTrusted: false;
  readonly licenseReviewComplete: false;
  readonly distributionAuthorized: false;
  readonly workerExecutionObserved: false;
  readonly loweringAuthorityMinted: false;
  readonly backendExecutionObserved: false;
  readonly releaseReady: false;
}

export type CppCuteBrowserFullDistributionReproducibilityResourceErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-EVIDENCE-INVALID"
  | "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-EVIDENCE-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-EVIDENCE-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-EVIDENCE-UNVERIFIED";

export class CppCuteBrowserFullDistributionReproducibilityResourceError
  extends Error {
  constructor(
    readonly code:
      CppCuteBrowserFullDistributionReproducibilityResourceErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name =
      "CppCuteBrowserFullDistributionReproducibilityResourceError";
  }
}

/** Returns a disposable copy of the exact checked-in observation bytes. */
export function cppCuteBrowserFullDistributionReproducibilityResourceBytes():
Uint8Array {
  return new Uint8Array(BUILTIN_RESOURCE_BYTES);
}

/**
 * Admits only the exact package observation, then independently binds every
 * deterministic output to the current build plan and package authorities.
 */
export async function
verifyCppCuteBrowserFullDistributionReproducibilityResource(
  value: Uint8Array,
): Promise<VerifiedCppCuteBrowserFullDistributionReproducibility> {
  let inspected: ReturnType<typeof inspectUnsharedPlainUint8Array>;
  try {
    inspected = inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-EVIDENCE-INVALID",
      "$bytes",
      "evidence must be one plain unshared Uint8Array",
      { cause },
    );
  }
  if (inspected.byteLength !==
      CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-EVIDENCE-RESOURCE-LIMIT",
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
      CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_SHA256) {
    mismatch(
      "$bytes.sha256",
      "evidence digest differs from its package identity",
    );
  }

  const resource =
    CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE;
  if (resource.verifierSourceRevision !==
        CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_OBSERVATION_SOURCE_REVISION ||
      resource.materializerSourceRevision !==
        CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_OBSERVATION_SOURCE_REVISION) {
    mismatch(
      "$.verifierSourceRevision",
      "observation source revision differs from the reviewed materializer and verifier revision",
    );
  }

  const [
    buildInputLock,
    headerDistribution,
    wasmReproducibility,
    workerBundle,
    diagnosticNormalization,
    runtimeAbi,
    semanticAdapter,
  ] = await Promise.all([
    decodeCppCuteBrowserBuildInputLock(
      cppCuteBrowserBuildInputLockResourceBytes(),
    ),
    verifyCppCuteBrowserHeaderDistributionReproducibilityResource(
      cppCuteBrowserHeaderDistributionReproducibilityResourceBytes(),
    ),
    verifyCppCuteBrowserReproducibilityResource(
      cppCuteBrowserReproducibilityResourceBytes(),
    ),
    verifyCppCuteBrowserWorkerBundle(),
    decodeCppCuteDiagnosticNormalization(
      cppCuteDiagnosticNormalizationResourceBytes(),
    ),
    decodeCppCuteBrowserRuntimeAbiManifest(
      cppCuteBrowserRuntimeAbiManifestResourceBytes(),
    ),
    decodeCppCuteSemanticAdapterManifest(
      cppCuteSemanticAdapterManifestResourceBytes(),
    ),
  ]);
  if (resource.buildInputLockId !== buildInputLock.lockId ||
      resource.buildInputLockResourceSha256 !==
        buildInputLock.resourceSha256) {
    mismatch(
      "$.buildInputLockId",
      "observation differs from the exact current package build-input lock",
    );
  }

  const lockRecord =
    unwrapPreparedCppCuteBrowserBuildInputLock(buildInputLock);
  assertExactOutputPlan(
    resource.deterministicOutputs,
    lockRecord.lock.body.recipe.distributedOutputPlan.outputs,
    resource.detachedEvidence,
  );
  assertDeterministicOutputProjection(
    resource.deterministicOutputs,
    headerDistribution.outputs,
    wasmReproducibility,
    inspectVerifiedCppCuteBrowserWorkerBundle(workerBundle),
    buildInputLock,
    diagnosticNormalization,
    runtimeAbi,
    semanticAdapter,
    resource.deterministicMetadata,
  );
  assertMetadataBindings(
    resource.deterministicMetadata,
    buildInputLock,
    headerDistribution,
    wasmReproducibility,
    inspectVerifiedCppCuteBrowserWorkerBundle(workerBundle),
    resource.detachedEvidence,
  );
  assertTotals(resource);

  const reproducibilityHash = await hash(canonicalJsonBytes({
    domain: REPRODUCIBILITY_HASH_DOMAIN,
    buildInputLockId: buildInputLock.lockId,
    buildInputLockResourceSha256: buildInputLock.resourceSha256,
    deterministicOutputs: resource.deterministicOutputs,
    detachedEvidence: resource.detachedEvidence,
  }), "$.reproducibilityId");
  const expectedReproducibilityId =
    `bg.cpp.browser-full-distribution-reproducibility.sha256.${reproducibilityHash}`;
  if (resource.reproducibilityId !== expectedReproducibilityId) {
    mismatch(
      "$.reproducibilityId",
      "reproducibility identity does not match the exact output observation",
    );
  }

  const authority = Object.freeze({
    authority:
      "package-pinned-full-distribution-reproducibility-only",
    resourceSha256:
      CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_SHA256,
    resourceByteLength:
      CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_RESOURCE_BYTE_LENGTH,
    observationVerifierSourceRevision: resource.verifierSourceRevision,
    materializerSourceRevision: resource.materializerSourceRevision,
    producerPolicyScope: resource.producerPolicyScope,
    buildInputLockId: resource.buildInputLockId,
    buildInputLockResourceSha256:
      resource.buildInputLockResourceSha256,
    reproducibilityId: resource.reproducibilityId,
    deterministicMetadata: resource.deterministicMetadata,
    deterministicOutputs: resource.deterministicOutputs,
    detachedEvidence: resource.detachedEvidence,
    outputCount: resource.totals.outputCount,
    deterministicSubjectCount:
      resource.totals.deterministicSubjectCount,
    detachedEvidenceCount: resource.totals.detachedEvidenceCount,
    firstByteLength: resource.totals.firstByteLength,
    secondByteLength: resource.totals.secondByteLength,
    ...resource.claims,
  }) as VerifiedCppCuteBrowserFullDistributionReproducibility;
  VERIFIED_RESOURCES.add(authority);
  return authority;
}

export function
requireVerifiedCppCuteBrowserFullDistributionReproducibility(
  authority: VerifiedCppCuteBrowserFullDistributionReproducibility,
): void {
  if (typeof authority !== "object" || authority === null ||
      !VERIFIED_RESOURCES.has(authority)) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-EVIDENCE-UNVERIFIED",
      "$authority",
      "expected verifier-issued full-distribution reproducibility authority",
    );
  }
}

function assertExactOutputPlan(
  outputs: readonly CppCuteBrowserFullDistributionReproducibilityOutputV1[],
  plan: readonly {
    readonly path: string;
    readonly role: string;
    readonly mediaType: string;
    readonly reproducibilityClass:
      "deterministic-subject" | "detached-evidence";
  }[],
  detached:
    typeof CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE.detachedEvidence,
): void {
  if (plan.length !== 25 || outputs.length !== 24) {
    mismatch("$.deterministicOutputs", "expected the closed 24+1 output plan");
  }
  const deterministicPlan = plan.filter(
    (output) => output.reproducibilityClass === "deterministic-subject",
  );
  const detachedPlan = plan.filter(
    (output) => output.reproducibilityClass === "detached-evidence",
  );
  if (deterministicPlan.length !== outputs.length ||
      detachedPlan.length !== 1) {
    mismatch(
      "$.deterministicOutputs",
      "build-lock reproducibility classes differ from the observation",
    );
  }
  for (const [index, output] of outputs.entries()) {
    const planned = deterministicPlan[index];
    if (planned === undefined ||
        output.outputPath !== planned.path ||
        output.role !== planned.role ||
        output.mediaType !== planned.mediaType) {
      mismatch(
        `$.deterministicOutputs[${index}]`,
        "output path, role, or media type differs from the exact build-lock plan",
      );
    }
    if (!SHA256.test(output.sha256) ||
        !WIRE_U64.test(output.byteLength)) {
      mismatch(
        `$.deterministicOutputs[${index}]`,
        "output identity is not canonical SHA-256 plus WireU64",
      );
    }
  }
  const plannedDetached = detachedPlan[0];
  if (plannedDetached === undefined ||
      detached.outputPath !== plannedDetached.path ||
      detached.role !== plannedDetached.role ||
      detached.mediaType !== plannedDetached.mediaType) {
    mismatch(
      "$.detachedEvidence",
      "detached output differs from the exact build-lock plan",
    );
  }
}

function assertDeterministicOutputProjection(
  outputs: readonly CppCuteBrowserFullDistributionReproducibilityOutputV1[],
  headerOutputs: readonly {
    readonly outputPath: string;
    readonly sha256: string;
    readonly byteLength: string;
  }[],
  wasm: {
    readonly wasmSha256: string;
    readonly wasmByteLength: number;
  },
  worker: {
    readonly sha256: string;
    readonly byteLength: number;
  },
  buildLock: {
    readonly resourceSha256: string;
    readonly resourceByteLength: number;
  },
  diagnostic: {
    readonly resourceSha256: string;
    readonly resourceByteLength: number;
  },
  runtimeAbi: {
    readonly resourceSha256: string;
    readonly resourceByteLength: number;
  },
  semanticAdapter: {
    readonly resourceSha256: string;
    readonly resourceByteLength: number;
  },
  metadata:
    typeof CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE.deterministicMetadata,
): void {
  const byPath = new Map(outputs.map((output) => [output.outputPath, output]));
  for (const header of headerOutputs) {
    assertOutput(
      byPath,
      header.outputPath,
      header.sha256,
      header.byteLength,
      "$.deterministicOutputs.headerDistribution",
    );
  }
  assertOutput(
    byPath,
    "assets/browsergrad-cpp-cute/clang-extractor.wasm",
    wasm.wasmSha256,
    String(wasm.wasmByteLength),
    "$.deterministicOutputs.clangExtractor",
  );
  assertOutput(
    byPath,
    "assets/browsergrad-cpp-cute/cpp-cute-browser-worker.mjs",
    worker.sha256,
    String(worker.byteLength),
    "$.deterministicOutputs.worker",
  );
  assertOutput(
    byPath,
    "assets/browsergrad-cpp-cute/build-input-lock.json",
    buildLock.resourceSha256,
    String(buildLock.resourceByteLength),
    "$.deterministicOutputs.buildInputLock",
  );
  assertOutput(
    byPath,
    "assets/browsergrad-cpp-cute/diagnostic-normalization.json",
    diagnostic.resourceSha256,
    String(diagnostic.resourceByteLength),
    "$.deterministicOutputs.diagnosticNormalization",
  );
  assertOutput(
    byPath,
    "assets/browsergrad-cpp-cute/runtime-abi-manifest.json",
    runtimeAbi.resourceSha256,
    String(runtimeAbi.resourceByteLength),
    "$.deterministicOutputs.runtimeAbi",
  );
  assertOutput(
    byPath,
    "assets/browsergrad-cpp-cute/semantic-adapter-manifest.json",
    semanticAdapter.resourceSha256,
    String(semanticAdapter.resourceByteLength),
    "$.deterministicOutputs.semanticAdapter",
  );
  assertOutput(
    byPath,
    "assets/browsergrad-cpp-cute/asset-manifest.json",
    metadata.assetManifestSha256,
    metadata.assetManifestByteLength,
    "$.deterministicOutputs.assetManifest",
  );
  if (byPath.size !== 24) {
    mismatch(
      "$.deterministicOutputs",
      "deterministic output paths are not unique",
    );
  }
}

function assertMetadataBindings(
  metadata:
    typeof CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE.deterministicMetadata,
  buildLock: {
    readonly lockId: string;
    readonly resourceSha256: string;
  },
  header: {
    readonly reproducibilityId: string;
    readonly outputVerificationId: string;
  },
  wasm: {
    readonly wasmSha256: string;
    readonly wasmByteLength: number;
  },
  worker: {
    readonly sha256: string;
  },
  detached:
    typeof CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE.detachedEvidence,
): void {
  if (!METADATA_ID.test(metadata.metadataId) ||
      metadata.profileId !==
        "browsergrad.compiler.cpp-cute.browser-clang@1" ||
      !SHA256.test(metadata.profileHash) ||
      !SHA256.test(metadata.compilationContractHash) ||
      !SHA256.test(metadata.profileSha256) ||
      !WIRE_U64.test(metadata.profileByteLength) ||
      !ASSET_MANIFEST_ID.test(metadata.assetManifestId) ||
      !SHA256.test(metadata.assetSetSha256)) {
    mismatch(
      "$.deterministicMetadata",
      "metadata identities are not canonical",
    );
  }
  const subjectMatch = BUILD_SUBJECT_ID.exec(metadata.buildSubjectId);
  if (subjectMatch?.[1] !== metadata.buildSubjectSha256 ||
      metadata.buildSubjectId !== detached.buildSubjectId ||
      metadata.buildSubjectSha256 !== detached.buildSubjectSha256) {
    mismatch(
      "$.deterministicMetadata.buildSubjectId",
      "metadata and detached evidence do not bind the same canonical build subject",
    );
  }
  if (metadata.wasmSha256 !== wasm.wasmSha256 ||
      metadata.wasmByteLength !== String(wasm.wasmByteLength) ||
      metadata.workerBundleSha256 !== worker.sha256 ||
      metadata.headerDistributionReproducibilityId !==
        header.reproducibilityId ||
      metadata.headerDistributionOutputVerificationId !==
        header.outputVerificationId) {
    mismatch(
      "$.deterministicMetadata",
      "metadata differs from current package Wasm, Worker, or header authorities",
    );
  }
  if (buildLock.lockId !==
        CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE.buildInputLockId ||
      buildLock.resourceSha256 !==
        CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE
          .buildInputLockResourceSha256) {
    mismatch(
      "$.buildInputLockId",
      "metadata verifier received a different current build lock",
    );
  }
}

function assertTotals(
  resource:
    typeof CPP_CUTE_BROWSER_FULL_DISTRIBUTION_REPRODUCIBILITY_V1_RESOURCE,
): void {
  const deterministicByteLength = resource.deterministicOutputs.reduce(
    (total, output) => total + BigInt(output.byteLength),
    0n,
  );
  const firstByteLength =
    deterministicByteLength + BigInt(resource.detachedEvidence.firstByteLength);
  const secondByteLength =
    deterministicByteLength + BigInt(resource.detachedEvidence.secondByteLength);
  if (resource.totals.outputCount !== 25 ||
      resource.totals.deterministicSubjectCount !== 24 ||
      resource.totals.detachedEvidenceCount !== 1 ||
      firstByteLength.toString() !== resource.totals.firstByteLength ||
      secondByteLength.toString() !== resource.totals.secondByteLength) {
    mismatch("$.totals", "recorded byte totals do not match the exact outputs");
  }
  if (!SHA256.test(resource.detachedEvidence.firstSha256) ||
      !SHA256.test(resource.detachedEvidence.secondSha256) ||
      !WIRE_U64.test(resource.detachedEvidence.firstByteLength) ||
      !WIRE_U64.test(resource.detachedEvidence.secondByteLength)) {
    mismatch(
      "$.detachedEvidence",
      "detached identities are not canonical SHA-256 plus WireU64",
    );
  }
}

function assertOutput(
  byPath: ReadonlyMap<
    string,
    CppCuteBrowserFullDistributionReproducibilityOutputV1
  >,
  outputPath: string,
  sha256: string,
  byteLength: string,
  path: string,
): void {
  const output = byPath.get(outputPath);
  if (output?.sha256 !== sha256 || output.byteLength !== byteLength) {
    mismatch(path, `output identity differs for ${outputPath}`);
  }
}

async function hash(bytes: Uint8Array, path: string): Promise<string> {
  try {
    return await sha256Hex(bytes);
  } catch (cause) {
    fail(
      "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-EVIDENCE-UNVERIFIED",
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
    "BG-COMPILER-CPP-CUTE-BROWSER-FULL-DISTRIBUTION-EVIDENCE-MISMATCH",
    path,
    message,
  );
}

function fail(
  code: CppCuteBrowserFullDistributionReproducibilityResourceErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteBrowserFullDistributionReproducibilityResourceError(
    code,
    path,
    message,
    options,
  );
}
