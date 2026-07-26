import {
  canonicalJsonBytes,
  hashCanonicalJson,
  sha256Hex,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
  cppCuteBrowserSourceAbi,
  deriveCppCuteBrowserAssetManifestId,
  deriveCppCuteBrowserAssetSetSha256,
  prepareCppCuteBrowserAssetManifest,
  type PreparedCppCuteBrowserAssetManifest,
  type CppCuteBrowserAssetManifestBodyV1,
  type CppCuteBrowserAssetManifestV1,
} from "../../../src/cpp_cute_browser_assets.js";
import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  type PreparedCppCuteBrowserBuildInputLock,
} from "../../../src/cpp_cute_browser_build_lock.js";
import {
  createCppCuteBrowserBuildProvenanceSigningRequest,
  verifyCppCuteBrowserBuildSignatureBinding,
} from "../../../src/cpp_cute_browser_build_provenance.js";
import {
  deriveCppCuteBrowserBuildSubjectIdentity,
  type CppCuteBrowserBuildProvenanceEnvelopeV1,
} from "../../../src/cpp_cute_browser_build_provenance_syntax.js";
import {
  createCppCuteBrowserCompileProfileInput,
  deriveCppCuteBrowserEmptySourceIncludeRootManifestSha256,
} from "../../../src/cpp_cute_browser_compile_profile.js";
import type {
  VerifiedCppCuteBrowserFullDistributionReproducibility,
} from "../../../src/cpp_cute_browser_full_distribution_reproducibility.js";
import {
  CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
} from "../../../src/cpp_cute_browser_runtime_abi.js";
import {
  CPP_CUTE_ATTESTATION_TRUST_STORE_MAJOR,
  CPP_CUTE_ATTESTATION_TRUST_STORE_MINOR,
  CPP_CUTE_FRONTEND_TRUST_STORE_SCHEMA,
  prepareCppCuteAttestationTrustStore,
  type CppCuteAttestationTrustStoreV1,
  type PreparedCppCuteAttestationTrustStore,
} from "../../../src/cpp_cute_frontend_provenance.js";
import {
  prepareCppCuteFrontendProfile,
  type CppCuteFrontendDependencyProfile,
  type PreparedCppCuteFrontendProfile,
} from "../../../src/cpp_cute_frontend_profile.js";
import {
  verifyCppCuteBrowserBuildProducer,
  type VerifiedCppCuteBrowserBuildProducer,
} from "../../../src/cpp_cute_browser_producer_trust.js";
import {
  admitCppCuteBrowserProducerTrustPolicy,
  type AdmittedCppCuteBrowserProducerTrustPolicy,
} from "../../../src/cpp_cute_browser_producer_trust_policy.js";
import {
  cppCuteBrowserReproducibilityResourceBytes,
  verifyCppCuteBrowserReproducibilityResource,
} from "../../../src/cpp_cute_browser_reproducibility.js";
import {
  CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256,
} from "../../../src/cpp_cute_semantic_adapter_manifest.js";
import {
  inspectVerifiedCppCuteBrowserWorkerBundle,
  verifyCppCuteBrowserWorkerBundle,
  type VerifiedCppCuteBrowserWorkerBundle,
} from "../../../src/cpp_cute_browser_worker_bundle.js";
import {
  cppCuteBrowserProducerTrustPolicyBytes,
} from "./cpp_cute_browser_producer_trust_fixtures.js";
import exactAssetManifestJson from
  "./resources/cpp_cute_browser_exact_asset_manifest.json"
  with { type: "json" };

const BUILDER_ID =
  "https://builders.browsergrad.dev/production-composition-test";

export interface CurrentPayloadExternallyRootedProducerFixture {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly assetManifest: PreparedCppCuteBrowserAssetManifest;
  readonly buildInputLock: PreparedCppCuteBrowserBuildInputLock;
  readonly workerBundle: VerifiedCppCuteBrowserWorkerBundle;
  readonly trustPolicy: AdmittedCppCuteBrowserProducerTrustPolicy;
  readonly trustStore: PreparedCppCuteAttestationTrustStore;
  readonly trustStoreInput: CppCuteAttestationTrustStoreV1;
  readonly privateKey: CryptoKey;
  readonly builderId: string;
  readonly keyId: string;
}

export async function createCurrentPayloadExternallyRootedProducer(
  fullDistribution:
    VerifiedCppCuteBrowserFullDistributionReproducibility,
): Promise<VerifiedCppCuteBrowserBuildProducer> {
  const fixture =
    await createCurrentPayloadExternallyRootedProducerFixture(
      fullDistribution,
    );
  const request = await createCppCuteBrowserBuildProvenanceSigningRequest({
    assetManifest: fixture.assetManifest,
    buildInputLock: fixture.buildInputLock,
    workerBundle: fixture.workerBundle,
    trustPolicy: fixture.trustPolicy,
    trustStore: fixture.trustStore,
    builderId: fixture.builderId,
    keyId: fixture.keyId,
  });
  const subtle = globalThis.crypto?.subtle;
  if (subtle === undefined) throw new Error("Web Crypto is required");
  const signature = new Uint8Array(await subtle.sign(
    { name: "ECDSA", hash: "SHA-256" },
    fixture.privateKey,
    Uint8Array.from(request.signingBytes).buffer,
  ));
  const envelope: CppCuteBrowserBuildProvenanceEnvelopeV1 = {
    payloadType: request.payloadType,
    payload: request.payload,
    signatures: [{
      keyid: fixture.keyId,
      sig: encodeBase64(signature),
    }],
  };
  const signatureBinding = await verifyCppCuteBrowserBuildSignatureBinding(
    envelope,
    {
      assetManifest: fixture.assetManifest,
      buildInputLock: fixture.buildInputLock,
      workerBundle: fixture.workerBundle,
      trustStore: fixture.trustStore,
    },
  );
  return await verifyCppCuteBrowserBuildProducer(
    signatureBinding,
    fixture.trustPolicy,
  );
}

export async function createCurrentPayloadExternallyRootedProducerFixture(
  fullDistribution:
    VerifiedCppCuteBrowserFullDistributionReproducibility,
): Promise<CurrentPayloadExternallyRootedProducerFixture> {
  const subtle = globalThis.crypto?.subtle;
  if (subtle === undefined) throw new Error("Web Crypto is required");
  const keyPair = await subtle.generateKey(
    { name: "ECDSA", namedCurve: "P-256" },
    true,
    ["sign", "verify"],
  );
  const spki = new Uint8Array(
    await subtle.exportKey("spki", keyPair.publicKey),
  );
  const keyId = `sha256:${await sha256Hex(spki)}`;
  const trustStoreInput: CppCuteAttestationTrustStoreV1 = {
    schema: CPP_CUTE_FRONTEND_TRUST_STORE_SCHEMA,
    version: {
      major: CPP_CUTE_ATTESTATION_TRUST_STORE_MAJOR,
      minor: CPP_CUTE_ATTESTATION_TRUST_STORE_MINOR,
    },
    keys: [{
      keyId,
      builderId: BUILDER_ID,
      algorithm: "ecdsa-p256-sha256",
      spkiDerBase64: encodeBase64(spki),
    }],
  };
  const trustStore =
    await prepareCppCuteAttestationTrustStore(trustStoreInput);
  const trustPolicy = await admitCppCuteBrowserProducerTrustPolicy(
    await cppCuteBrowserProducerTrustPolicyBytes({
      trustStoreSha256: trustStore.trustStoreHash,
      builderIds: [BUILDER_ID],
      keyIds: [keyId],
    }),
  );
  const [buildInputLock, wasmReproducibility, workerBundle] =
    await Promise.all([
      decodeCppCuteBrowserBuildInputLock(
        cppCuteBrowserBuildInputLockResourceBytes(),
      ),
      verifyCppCuteBrowserReproducibilityResource(
        cppCuteBrowserReproducibilityResourceBytes(),
      ),
      verifyCppCuteBrowserWorkerBundle(),
    ]);
  const originalManifest = exactAssetManifest();
  await requireExactFixture(originalManifest, fullDistribution);
  const worker = inspectVerifiedCppCuteBrowserWorkerBundle(workerBundle);
  const policy = {
    predicateType: CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
    trustStoreSha256: trustStore.trustStoreHash,
    builderIds: [BUILDER_ID],
  } as const;
  const assetSetSha256 = await deriveCppCuteBrowserAssetSetSha256({
    sourceAbiSha256: originalManifest.body.sourceAbiSha256,
    dependencyIds: originalManifest.body.dependencyIds,
    buildSubjectIds: originalManifest.body.buildSubjectIds,
    buildProvenancePolicy: policy,
    mountedVirtualRoots: originalManifest.body.mountedVirtualRoots,
    assets: originalManifest.body.assets,
  });
  const profile = await prepareCppCuteFrontendProfile(
    createCppCuteBrowserCompileProfileInput({
      assetSetSha256,
      buildProvenanceLockSha256: buildInputLock.resourceSha256,
      extractorWasmSha256: wasmReproducibility.wasmSha256,
      runtimeAbiManifestSha256:
        CPP_CUTE_BROWSER_RUNTIME_ABI_V1_RESOURCE_SHA256,
      semanticAdapterManifestSha256:
        CPP_CUTE_SEMANTIC_ADAPTER_MANIFEST_V1_RESOURCE_SHA256,
      sourceRootManifestSha256:
        await deriveCppCuteBrowserEmptySourceIncludeRootManifestSha256(),
      workerModuleSha256: worker.sha256,
      workerModuleByteLength: worker.byteLength,
      headerContentSets: {
        clangResource:
          originalManifest.body.sourceAbi.toolchain.compiler
            .resourceDirectorySha256,
        cuda: dependencyHeaderSet(originalManifest, "cuda"),
        cutlass: dependencyHeaderSet(originalManifest, "cutlass"),
        cxxStdlib: dependencyHeaderSet(originalManifest, "cxx-stdlib"),
        linuxSysroot:
          dependencyHeaderSet(originalManifest, "linux-sysroot"),
      },
    }),
  );
  const sourceAbi = cppCuteBrowserSourceAbi(profile);
  if (await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.browser-source-abi.v1",
    sourceAbi,
  }) !== originalManifest.body.sourceAbiSha256) {
    throw new Error("exact fixture source ABI drifted");
  }
  const body: CppCuteBrowserAssetManifestBodyV1 = {
    ...originalManifest.body,
    profileHash: profile.profileHash,
    assetSetSha256,
    buildProvenancePolicy: policy,
  };
  const assetManifest = await prepareCppCuteBrowserAssetManifest({
    ...originalManifest,
    manifestId: await deriveCppCuteBrowserAssetManifestId(body),
    body,
  }, profile);
  const buildSubject = await deriveCppCuteBrowserBuildSubjectIdentity({
    assetManifest,
    buildInputLock,
    workerBundle,
  });
  if (buildSubject.buildSubjectId !==
        fullDistribution.deterministicMetadata.buildSubjectId) {
    throw new Error(
      "test trust-root migration changed the exact payload build subject",
    );
  }
  return {
    profile,
    assetManifest,
    buildInputLock,
    workerBundle,
    trustPolicy,
    trustStore,
    trustStoreInput,
    privateKey: keyPair.privateKey,
    builderId: BUILDER_ID,
    keyId,
  };
}

function exactAssetManifest(): CppCuteBrowserAssetManifestV1 {
  return structuredClone(exactAssetManifestJson) as unknown as
    CppCuteBrowserAssetManifestV1;
}

async function requireExactFixture(
  manifest: CppCuteBrowserAssetManifestV1,
  fullDistribution:
    VerifiedCppCuteBrowserFullDistributionReproducibility,
): Promise<void> {
  const metadata = fullDistribution.deterministicMetadata;
  if (manifest.manifestId !== metadata.assetManifestId ||
      manifest.body.profileHash !== metadata.profileHash ||
      manifest.body.assetSetSha256 !== metadata.assetSetSha256 ||
      manifest.body.buildSubjectIds.length !== 1 ||
      manifest.body.buildSubjectIds[0] !== metadata.buildSubjectId ||
      await sha256Hex(canonicalJsonBytes(manifest)) !==
        metadata.assetManifestSha256) {
    throw new Error(
      "exact asset-manifest fixture differs from package reproducibility",
    );
  }
}

function dependencyHeaderSet(
  manifest: CppCuteBrowserAssetManifestV1,
  dependencyId: string,
): string {
  const dependency = manifest.body.sourceAbi.toolchain.dependencies.find(
    (candidate: CppCuteFrontendDependencyProfile) =>
      candidate.dependencyId === dependencyId,
  );
  if (dependency === undefined) {
    throw new Error(`missing exact dependency ${dependencyId}`);
  }
  return dependency.headerSetSha256;
}

function encodeBase64(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes));
}
