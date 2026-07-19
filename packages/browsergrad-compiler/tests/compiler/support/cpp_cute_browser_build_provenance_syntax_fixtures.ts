import {
  prepareCppCuteBrowserAssetManifest,
  type PreparedCppCuteBrowserAssetManifest,
} from "../../../src/cpp_cute_browser_assets.js";
import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  type PreparedCppCuteBrowserBuildInputLock,
} from "../../../src/cpp_cute_browser_build_lock.js";
import {
  CPP_CUTE_BROWSER_BUILD_DSSE_PAYLOAD_TYPE,
  CPP_CUTE_BROWSER_BUILD_IN_TOTO_STATEMENT_TYPE,
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
  CPP_CUTE_BROWSER_BUILD_TYPE,
  cppCuteBrowserBuildProvenancePayloadBase64,
  deriveCppCuteBrowserBuildSubjectIdentity,
  type CppCuteBrowserBuildProvenanceEnvelopeV1,
  type CppCuteBrowserBuildProvenanceStatementV1,
  type CppCuteBrowserBuildSubjectIdentity,
} from "../../../src/cpp_cute_browser_build_provenance_syntax.js";
import {
  inspectVerifiedCppCuteBrowserWorkerBundle,
  verifyCppCuteBrowserWorkerBundle,
  type VerifiedCppCuteBrowserWorkerBundle,
} from "../../../src/cpp_cute_browser_worker_bundle.js";
import type { PreparedCppCuteFrontendProfile } from "../../../src/cpp_cute_frontend_profile.js";
import { createCppCuteBrowserAssetFixture } from "./cpp_cute_browser_asset_fixtures.js";

export const CPP_CUTE_BROWSER_BUILD_SYNTAX_FIXTURE_BUILDER_ID =
  "https://builders.browsergrad.dev/cpp-cute-browser-test";

export interface CppCuteBrowserBuildProvenanceSyntaxFixture {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly assetManifest: PreparedCppCuteBrowserAssetManifest;
  readonly buildInputLock: PreparedCppCuteBrowserBuildInputLock;
  readonly workerBundle: VerifiedCppCuteBrowserWorkerBundle;
  readonly buildSubject: CppCuteBrowserBuildSubjectIdentity;
  readonly statement: CppCuteBrowserBuildProvenanceStatementV1;
  readonly envelope: CppCuteBrowserBuildProvenanceEnvelopeV1;
}

/** Creates syntactically valid but deliberately unsigned/untrusted test data. */
export async function createCppCuteBrowserBuildProvenanceSyntaxFixture():
Promise<CppCuteBrowserBuildProvenanceSyntaxFixture> {
  const buildInputLock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
  const assetFixture = await createCppCuteBrowserAssetFixture({
    profile: { buildProvenanceLockSha256: buildInputLock.resourceSha256 },
  });
  const assetManifest = await prepareCppCuteBrowserAssetManifest(
    assetFixture.input,
    assetFixture.profile,
  );
  const workerBundle = await verifyCppCuteBrowserWorkerBundle();
  const worker = inspectVerifiedCppCuteBrowserWorkerBundle(workerBundle);
  const buildSubject = await deriveCppCuteBrowserBuildSubjectIdentity({
    assetManifest,
    buildInputLock,
    workerBundle,
  });
  const statement: CppCuteBrowserBuildProvenanceStatementV1 = {
    _type: CPP_CUTE_BROWSER_BUILD_IN_TOTO_STATEMENT_TYPE,
    subject: [{
      name: buildSubject.buildSubjectId,
      digest: { sha256: buildSubject.buildSubjectSha256 },
    }],
    predicateType: CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
    predicate: {
      builderId: CPP_CUTE_BROWSER_BUILD_SYNTAX_FIXTURE_BUILDER_ID,
      buildType: CPP_CUTE_BROWSER_BUILD_TYPE,
      buildSubject: {
        buildSubjectId: buildSubject.buildSubjectId,
        buildSubjectSha256: buildSubject.buildSubjectSha256,
      },
      profile: {
        profileId: assetFixture.profile.profileId,
        profileHash: assetFixture.profile.profileHash,
        compilationContractHash: assetFixture.profile.compilationContractHash,
      },
      assetManifest: {
        manifestId: assetManifest.manifestId,
        manifestSha256: assetManifest.manifestSha256,
        manifestByteLength: assetManifest.manifestByteLength,
        assetSetSha256: assetManifest.assetSetSha256,
      },
      buildInputLock: {
        lockId: buildInputLock.lockId,
        resourceSha256: buildInputLock.resourceSha256,
        recipeSha256: buildInputLock.recipeSha256,
      },
      workerBundle: {
        bundleId: worker.bundleId,
        sha256: worker.sha256,
        byteLength: worker.byteLength,
        factorySha256: worker.factorySha256,
        factoryByteLength: worker.factoryByteLength,
      },
      authorityLimits: {
        fullDistributedOutputSetReproducible: false,
        licenseReviewComplete: false,
        distributionAuthorized: false,
        releaseReady: false,
      },
    },
  };
  const envelope: CppCuteBrowserBuildProvenanceEnvelopeV1 = {
    payloadType: CPP_CUTE_BROWSER_BUILD_DSSE_PAYLOAD_TYPE,
    payload: cppCuteBrowserBuildProvenancePayloadBase64(statement),
    signatures: [{
      keyid: `sha256:${"1".repeat(64)}`,
      sig: encodeBase64(new Uint8Array(64)),
    }],
  };
  return {
    profile: assetFixture.profile,
    assetManifest,
    buildInputLock,
    workerBundle,
    buildSubject,
    statement,
    envelope,
  };
}

export function encodeCppCuteBrowserBuildSyntaxBase64(bytes: Uint8Array): string {
  return encodeBase64(bytes);
}

function encodeBase64(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes));
}
