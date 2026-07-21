import {
  canonicalJsonBytes,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
} from "../../../src/cpp_cute_browser_assets.js";
import {
  CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MAJOR,
  CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MINOR,
  CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_SCHEMA,
  deriveCppCuteBrowserProducerTrustPolicyId,
  type CppCuteBrowserProducerTrustPolicyProjectionV1,
  type CppCuteBrowserProducerTrustPolicyV1,
} from "../../../src/cpp_cute_browser_producer_trust_policy.js";
import {
  createSignedCppCuteBrowserBuildProvenanceFixture,
  type SignedCppCuteBrowserBuildProvenanceFixture,
} from "./cpp_cute_browser_build_provenance_syntax_fixtures.js";

export interface CppCuteBrowserProducerTrustFixture {
  readonly build: SignedCppCuteBrowserBuildProvenanceFixture;
  readonly policy: CppCuteBrowserProducerTrustPolicyV1;
  readonly policyBytes: Uint8Array;
}

export async function createCppCuteBrowserProducerTrustFixture():
Promise<CppCuteBrowserProducerTrustFixture> {
  const build = await createSignedCppCuteBrowserBuildProvenanceFixture();
  const policyBytes = await cppCuteBrowserProducerTrustPolicyBytes({
    trustStoreSha256: build.trustStore.trustStoreHash,
    builderIds: [build.statement.predicate.builderId],
    keyIds: [build.envelope.signatures[0].keyid],
  });
  return {
    build,
    policy: JSON.parse(new TextDecoder().decode(policyBytes)) as
      CppCuteBrowserProducerTrustPolicyV1,
    policyBytes,
  };
}

export async function cppCuteBrowserProducerTrustPolicyBytes(input: {
  readonly trustStoreSha256: string;
  readonly builderIds: readonly string[];
  readonly keyIds: readonly string[];
  readonly version?: { readonly major: number; readonly minor: number };
  readonly predicateType?: string;
  readonly policyId?: string;
  readonly extra?: Readonly<Record<string, unknown>>;
}): Promise<Uint8Array> {
  const projection = {
    schema: CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_SCHEMA,
    version: input.version ?? {
      major: CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MAJOR,
      minor: CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MINOR,
    },
    predicateType: input.predicateType ??
      CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE,
    trustStoreSha256: input.trustStoreSha256,
    builderIds: [...input.builderIds],
    keyIds: [...input.keyIds],
  };
  let policyId = input.policyId;
  if (policyId === undefined &&
      projection.version.major === CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MAJOR &&
      projection.version.minor === CPP_CUTE_BROWSER_PRODUCER_TRUST_POLICY_MINOR &&
      projection.predicateType === CPP_CUTE_BROWSER_BUILD_PROVENANCE_PREDICATE_TYPE) {
    policyId = await deriveCppCuteBrowserProducerTrustPolicyId(
      projection as CppCuteBrowserProducerTrustPolicyProjectionV1,
    );
  }
  return canonicalJsonBytes({
    ...projection,
    policyId: policyId ?? `bg.cpp.browser-producer-trust-policy.sha256.${"0".repeat(64)}`,
    ...input.extra,
  });
}
