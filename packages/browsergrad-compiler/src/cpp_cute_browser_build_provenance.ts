import {
  hashCanonicalJson,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapPreparedCppCuteBrowserAssetManifest,
  type PreparedCppCuteBrowserAssetManifest,
} from "./cpp_cute_browser_assets.js";
import {
  unwrapPreparedCppCuteBrowserBuildInputLock,
  type PreparedCppCuteBrowserBuildInputLock,
} from "./cpp_cute_browser_build_lock.js";
import {
  cppCuteBrowserBuildProvenanceDsseSigningBytes,
  decodeUntrustedCppCuteBrowserBuildProvenanceSyntax,
  deriveCppCuteBrowserBuildSubjectIdentity,
  type CppCuteBrowserBuildProvenanceEnvelopeV1,
  type CppCuteBrowserBuildProvenanceStatementV1,
  type CppCuteBrowserBuildSubjectIdentity,
} from "./cpp_cute_browser_build_provenance_syntax.js";
import {
  CppCuteFrontendProvenanceError,
  verifyCppCutePreparedAttestationSignature,
  type PreparedCppCuteAttestationTrustStore,
} from "./cpp_cute_frontend_provenance.js";
import {
  unwrapPreparedCppCuteBrowserFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";
import {
  inspectVerifiedCppCuteBrowserWorkerBundle,
  type VerifiedCppCuteBrowserWorkerBundle,
} from "./cpp_cute_browser_worker_bundle.js";

const VERIFIED_BUILD_SIGNATURE_BINDINGS = new WeakMap<object, VerifiedCppCuteBrowserBuildSignatureBindingRecord>();

declare const verifiedCppCuteBrowserBuildSignatureBindingBrand: unique symbol;

/**
 * Manifest-policy signature/build-subject binding. This is deliberately named
 * as a signature binding rather than provenance authority. Until a separate
 * package-owned trust-root authority admits the manifest policy, this does not
 * prove producer trust, acquired asset bytes, legal approval, complete
 * reproducibility, Worker execution, or release readiness.
 */
export interface VerifiedCppCuteBrowserBuildSignatureBinding {
  readonly [verifiedCppCuteBrowserBuildSignatureBindingBrand]: true;
  readonly buildSubjectId: string;
  readonly buildSubjectSha256: string;
  readonly statementSha256: string;
  readonly evidenceSha256: string;
  readonly builderId: string;
  readonly keyId: string;
  readonly trustStoreSha256: string;
  readonly profileHash: string;
  readonly manifestId: string;
  readonly manifestSha256: string;
  readonly manifestByteLength: WireU64;
  readonly assetSetSha256: string;
  readonly buildInputLockResourceSha256: string;
  readonly workerBundleSha256: string;
  readonly signatureVerified: true;
  readonly manifestSignaturePolicyMatched: true;
  readonly producerTrusted: false;
  readonly buildSubjectBound: true;
  readonly exactAssetBytesVerified: false;
  readonly fullDistributedOutputSetReproducible: false;
  readonly licenseReviewComplete: false;
  readonly distributionAuthorized: false;
  readonly workerExecutionObserved: false;
  readonly releaseReady: false;
}

export interface VerifiedCppCuteBrowserBuildSignatureBindingRecord {
  readonly envelope: CppCuteBrowserBuildProvenanceEnvelopeV1;
  readonly statement: CppCuteBrowserBuildProvenanceStatementV1;
  readonly buildSubject: CppCuteBrowserBuildSubjectIdentity;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly assetManifest: PreparedCppCuteBrowserAssetManifest;
  readonly buildInputLock: PreparedCppCuteBrowserBuildInputLock;
  readonly workerBundle: VerifiedCppCuteBrowserWorkerBundle;
  readonly trustStore: PreparedCppCuteAttestationTrustStore;
}

export interface VerifyCppCuteBrowserBuildProvenanceRequest {
  readonly assetManifest: PreparedCppCuteBrowserAssetManifest;
  readonly buildInputLock: PreparedCppCuteBrowserBuildInputLock;
  readonly workerBundle: VerifiedCppCuteBrowserWorkerBundle;
  readonly trustStore: PreparedCppCuteAttestationTrustStore;
  readonly signal?: AbortSignal;
}

export type CppCuteBrowserBuildProvenanceErrorCode =
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-BINDING"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-POLICY"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-SIGNATURE"
  | "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-UNVERIFIED";

export class CppCuteBrowserBuildProvenanceError extends Error {
  constructor(
    readonly code: CppCuteBrowserBuildProvenanceErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteBrowserBuildProvenanceError";
  }
}

export async function verifyCppCuteBrowserBuildSignatureBinding(
  value: unknown,
  request: VerifyCppCuteBrowserBuildProvenanceRequest,
): Promise<VerifiedCppCuteBrowserBuildSignatureBinding> {
  const assetManifest = request.assetManifest;
  const buildInputLock = request.buildInputLock;
  const workerBundle = request.workerBundle;
  const trustStore = request.trustStore;
  const signal = request.signal;
  throwIfAborted(signal);
  const decoded = decodeUntrustedCppCuteBrowserBuildProvenanceSyntax(
    value,
    signal === undefined ? {} : { signal },
  );
  const manifestRecord = unwrapPreparedCppCuteBrowserAssetManifest(assetManifest);
  const profileRecord = unwrapPreparedCppCuteBrowserFrontendProfile(manifestRecord.profile);
  unwrapPreparedCppCuteBrowserBuildInputLock(buildInputLock);
  const lock = buildInputLock;
  const worker = inspectVerifiedCppCuteBrowserWorkerBundle(workerBundle);
  const buildSubject = await deriveCppCuteBrowserBuildSubjectIdentity({
    assetManifest,
    buildInputLock,
    workerBundle,
    ...(signal === undefined ? {} : { signal }),
  });
  throwIfAborted(signal);

  const profile = profileRecord.profile;
  const deployment = profile.deployment;
  const manifest = manifestRecord.manifest;
  const provenancePolicy = manifest.body.buildProvenancePolicy;
  const predicate = decoded.statement.predicate;
  const signature = decoded.envelope.signatures[0];

  if (deployment.buildProvenanceLockSha256 !== lock.resourceSha256) {
    binding("$.payload.predicate.buildInputLock.resourceSha256", "profile build-input lock differs from provenance input");
  }
  if (deployment.worker.moduleSha256 !== worker.sha256 ||
      deployment.worker.moduleByteLength !== worker.byteLength) {
    binding("$.payload.predicate.workerBundle", "profile Worker module differs from the verified package bundle");
  }
  if (manifest.body.buildSubjectIds.length !== 1 ||
      manifest.body.buildSubjectIds[0] !== buildSubject.buildSubjectId) {
    binding("$.assetManifest.body.buildSubjectIds", "asset manifest must reference exactly the derived build subject");
  }

  equal(decoded.statement.subject[0].name, buildSubject.buildSubjectId, "$.payload.subject[0].name");
  equal(decoded.statement.subject[0].digest.sha256, buildSubject.buildSubjectSha256, "$.payload.subject[0].digest.sha256");
  equal(predicate.buildSubject.buildSubjectId, buildSubject.buildSubjectId, "$.payload.predicate.buildSubject.buildSubjectId");
  equal(predicate.buildSubject.buildSubjectSha256, buildSubject.buildSubjectSha256, "$.payload.predicate.buildSubject.buildSubjectSha256");
  equal(predicate.profile.profileId, profile.profileId, "$.payload.predicate.profile.profileId");
  equal(predicate.profile.profileHash, profileRecord.profileHash, "$.payload.predicate.profile.profileHash");
  equal(
    predicate.profile.compilationContractHash,
    profileRecord.compilationContractHash,
    "$.payload.predicate.profile.compilationContractHash",
  );
  equal(predicate.assetManifest.manifestId, assetManifest.manifestId, "$.payload.predicate.assetManifest.manifestId");
  equal(predicate.assetManifest.manifestSha256, assetManifest.manifestSha256, "$.payload.predicate.assetManifest.manifestSha256");
  equal(predicate.assetManifest.manifestByteLength, assetManifest.manifestByteLength, "$.payload.predicate.assetManifest.manifestByteLength");
  equal(predicate.assetManifest.assetSetSha256, assetManifest.assetSetSha256, "$.payload.predicate.assetManifest.assetSetSha256");
  equal(predicate.buildInputLock.lockId, lock.lockId, "$.payload.predicate.buildInputLock.lockId");
  equal(predicate.buildInputLock.resourceSha256, lock.resourceSha256, "$.payload.predicate.buildInputLock.resourceSha256");
  equal(predicate.buildInputLock.recipeSha256, lock.recipeSha256, "$.payload.predicate.buildInputLock.recipeSha256");
  equal(predicate.workerBundle.bundleId, worker.bundleId, "$.payload.predicate.workerBundle.bundleId");
  equal(predicate.workerBundle.sha256, worker.sha256, "$.payload.predicate.workerBundle.sha256");
  equal(predicate.workerBundle.byteLength, worker.byteLength, "$.payload.predicate.workerBundle.byteLength");
  equal(predicate.workerBundle.factorySha256, worker.factorySha256, "$.payload.predicate.workerBundle.factorySha256");
  equal(predicate.workerBundle.factoryByteLength, worker.factoryByteLength, "$.payload.predicate.workerBundle.factoryByteLength");

  try {
    await verifyCppCutePreparedAttestationSignature({
      trustStore,
      expectedTrustStoreHash: provenancePolicy.trustStoreSha256,
      allowlistedBuilderIds: provenancePolicy.builderIds,
      builderId: predicate.builderId,
      keyId: signature.keyid,
      signatureBase64: signature.sig,
      signingBytes: cppCuteBrowserBuildProvenanceDsseSigningBytes(decoded.statement),
      ...(signal === undefined ? {} : { signal }),
    });
  } catch (cause) {
    if (cause instanceof CppCuteFrontendProvenanceError) {
      const code = cause.code === "BG-COMPILER-CPP-CUTE-PROVENANCE-CANCELLED"
        ? "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-CANCELLED"
        : cause.code === "BG-COMPILER-CPP-CUTE-PROVENANCE-POLICY-MISMATCH"
          ? "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-POLICY"
          : "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-SIGNATURE";
      throw new CppCuteBrowserBuildProvenanceError(
        code,
        cause.path,
        "browser build attestation failed profile-pinned signature verification",
        { cause },
      );
    }
    throw cause;
  }
  throwIfAborted(signal);
  const statementSha256 = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-build-provenance-statement.v1",
    statement: decoded.statement,
  });
  const evidenceSha256 = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-build-provenance-evidence.v1",
    statementSha256,
    buildSubjectId: buildSubject.buildSubjectId,
    profileHash: profileRecord.profileHash,
    manifestId: assetManifest.manifestId,
    manifestSha256: assetManifest.manifestSha256,
    buildInputLockResourceSha256: lock.resourceSha256,
    workerBundleSha256: worker.sha256,
    trustStoreSha256: provenancePolicy.trustStoreSha256,
    keyId: signature.keyid,
    builderId: predicate.builderId,
  });
  throwIfAborted(signal);
  const verified = Object.freeze({
    buildSubjectId: buildSubject.buildSubjectId,
    buildSubjectSha256: buildSubject.buildSubjectSha256,
    statementSha256,
    evidenceSha256,
    builderId: predicate.builderId,
    keyId: signature.keyid,
    trustStoreSha256: provenancePolicy.trustStoreSha256,
    profileHash: profileRecord.profileHash,
    manifestId: assetManifest.manifestId,
    manifestSha256: assetManifest.manifestSha256,
    manifestByteLength: assetManifest.manifestByteLength,
    assetSetSha256: assetManifest.assetSetSha256,
    buildInputLockResourceSha256: lock.resourceSha256,
    workerBundleSha256: worker.sha256,
    signatureVerified: true,
    manifestSignaturePolicyMatched: true,
    producerTrusted: false,
    buildSubjectBound: true,
    exactAssetBytesVerified: false,
    fullDistributedOutputSetReproducible: false,
    licenseReviewComplete: false,
    distributionAuthorized: false,
    workerExecutionObserved: false,
    releaseReady: false,
  }) as VerifiedCppCuteBrowserBuildSignatureBinding;
  VERIFIED_BUILD_SIGNATURE_BINDINGS.set(verified, Object.freeze({
    envelope: decoded.envelope,
    statement: decoded.statement,
    buildSubject,
    profile: manifestRecord.profile,
    assetManifest,
    buildInputLock,
    workerBundle,
    trustStore,
  }));
  return verified;
}

export function unwrapVerifiedCppCuteBrowserBuildSignatureBinding(
  verified: VerifiedCppCuteBrowserBuildSignatureBinding,
): VerifiedCppCuteBrowserBuildSignatureBindingRecord {
  if (typeof verified !== "object" || verified === null) unverified();
  const record = VERIFIED_BUILD_SIGNATURE_BINDINGS.get(verified as object);
  if (record === undefined) unverified();
  return record;
}

function equal(actual: string | number, expected: string | number, path: string): void {
  if (actual !== expected) binding(path, "provenance field differs from the exact opaque input authority");
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    throw new CppCuteBrowserBuildProvenanceError(
      "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-CANCELLED",
      "$.signal",
      "browser build provenance verification was cancelled",
    );
  }
}

function binding(path: string, message: string): never {
  throw new CppCuteBrowserBuildProvenanceError(
    "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-BINDING",
    path,
    message,
  );
}

function unverified(): never {
  throw new CppCuteBrowserBuildProvenanceError(
    "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-PROVENANCE-UNVERIFIED",
    "$",
    "browser build provenance must come from the verified opaque authority",
  );
}
