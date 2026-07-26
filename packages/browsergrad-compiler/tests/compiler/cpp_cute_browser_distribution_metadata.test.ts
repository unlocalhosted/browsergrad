import { canonicalJsonBytes } from
  "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";

import {
  copyPreparedCppCuteBrowserDistributionAssetManifestBytes,
  copyPreparedCppCuteBrowserDistributionProfileBytes,
  CppCuteBrowserDistributionMetadataError,
  prepareCppCuteBrowserDistributionMetadata,
  unwrapPreparedCppCuteBrowserDistributionMetadata,
  type CppCuteBrowserDistributionPackIncludeRootId,
  type CppCuteBrowserDistributionPackInput,
} from "../../src/cpp_cute_browser_distribution_metadata.js";
import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
} from "../../src/cpp_cute_browser_build_lock.js";
import {
  unwrapPreparedCppCuteBrowserFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import {
  admitCppCuteBrowserProducerTrustPolicy,
} from "../../src/cpp_cute_browser_producer_trust_policy.js";
import {
  cppCuteBrowserReproducibilityResourceBytes,
  verifyCppCuteBrowserReproducibilityResource,
} from "../../src/cpp_cute_browser_reproducibility.js";
import {
  encodeCppCuteBrowserVfsPack,
  inspectCppCuteBrowserVfsPack,
} from "../../src/cpp_cute_browser_vfs_pack.js";
import {
  verifyCppCuteBrowserWorkerBundle,
} from "../../src/cpp_cute_browser_worker_bundle.js";
import {
  cppCuteBrowserProducerTrustPolicyBytes,
} from "./support/cpp_cute_browser_producer_trust_fixtures.js";

const PACK_BINDINGS = [
  {
    includeRootId: "clang-resource",
    outputPath:
      "assets/browsergrad-cpp-cute/clang-resource.headers.bgvfs",
  },
  {
    includeRootId: "cuda",
    outputPath:
      "assets/browsergrad-cpp-cute/cuda-12.6.3.headers.bgvfs",
  },
  {
    includeRootId: "cutlass",
    outputPath:
      "assets/browsergrad-cpp-cute/cutlass-3.7.0.headers.bgvfs",
  },
  {
    includeRootId: "cxx-stdlib",
    outputPath:
      "assets/browsergrad-cpp-cute/libcxx-22.1.8.headers.bgvfs",
  },
  {
    includeRootId: "linux-sysroot",
    outputPath:
      "assets/browsergrad-cpp-cute/linux-sysroot.headers.bgvfs",
  },
] as const;

describe("browser C++/CuTe distribution metadata", () => {
  it("prepares one deterministic profile, manifest, and cycle-free build subject", async () => {
    const input = await metadataInput();
    const first = await prepareCppCuteBrowserDistributionMetadata(input);
    const second = await prepareCppCuteBrowserDistributionMetadata(input);

    expect(first).toMatchObject({
      schema:
        "browsergrad.compiler.cpp-cute.browser-distribution-metadata",
      version: 1,
      authority: "deterministic-browser-distribution-metadata-only",
      profileId: "browsergrad.compiler.cpp-cute.browser-clang@1",
      packCount: 5,
      profilePrepared: true,
      assetManifestPrepared: true,
      cycleFreeBuildSubjectVerified: true,
      exactOutputFilesVerified: false,
      producerTrusted: false,
      distributionAuthorized: false,
      releaseReady: false,
    });
    expect(first.metadataId).toBe(second.metadataId);
    expect(first.buildSubjectId).toBe(second.buildSubjectId);
    expect(first.assetManifestSha256).toBe(second.assetManifestSha256);
    expect(Object.isFrozen(first)).toBe(true);

    const record = unwrapPreparedCppCuteBrowserDistributionMetadata(first);
    expect(record.buildSubject.buildSubjectId).toBe(first.buildSubjectId);
    expect(record.assetManifest.manifestId).toBe(first.assetManifestId);
    expect(copyPreparedCppCuteBrowserDistributionProfileBytes(first))
      .toEqual(canonicalJsonBytes(
        unwrapPreparedCppCuteBrowserFrontendProfile(record.profile).profile,
      ));
    expect(
      copyPreparedCppCuteBrowserDistributionAssetManifestBytes(first)
        .byteLength,
    ).toBe(Number(first.assetManifestByteLength));

    const profileCopy =
      copyPreparedCppCuteBrowserDistributionProfileBytes(first);
    profileCopy.fill(0);
    expect(copyPreparedCppCuteBrowserDistributionProfileBytes(first)[0])
      .not.toBe(0);
    expect(() =>
      unwrapPreparedCppCuteBrowserDistributionMetadata({
        ...first,
      } as never),
    ).toThrow(CppCuteBrowserDistributionMetadataError);
  });

  it("rejects path drift, duplicate packs, and forged pack authority", async () => {
    const input = await metadataInput();
    await expect(prepareCppCuteBrowserDistributionMetadata({
      ...input,
      packs: input.packs.map((pack, index) => index === 0
        ? { ...pack, outputPath: "assets/browsergrad-cpp-cute/drift.bgvfs" }
        : pack),
    })).rejects.toThrow("binding differs from the distribution contract");

    await expect(prepareCppCuteBrowserDistributionMetadata({
      ...input,
      packs: input.packs.map((pack, index) => index === 1
        ? {
            ...pack,
            includeRootId: input.packs[0]?.includeRootId ??
              "clang-resource",
            outputPath: input.packs[0]?.outputPath ?? "",
          } as CppCuteBrowserDistributionPackInput
        : pack),
    })).rejects.toThrow("include roots must be unique");

    await expect(prepareCppCuteBrowserDistributionMetadata({
      ...input,
      packs: input.packs.map((pack, index) => index === 0
        ? { ...pack, pack: { ...pack.pack } as never }
        : pack),
    })).rejects.toThrow("opaque inspected VFS pack");
  });

  it("binds producer policy identity into the manifest and build subject", async () => {
    const input = await metadataInput();
    const first = await prepareCppCuteBrowserDistributionMetadata(input);
    const policyBytes = await cppCuteBrowserProducerTrustPolicyBytes({
      trustStoreSha256: "b".repeat(64),
      builderIds: ["https://builders.browsergrad.dev/alternate"],
      keyIds: [`sha256:${"c".repeat(64)}`],
    });
    const second = await prepareCppCuteBrowserDistributionMetadata({
      ...input,
      producerTrustPolicy:
        await admitCppCuteBrowserProducerTrustPolicy(policyBytes),
    });

    expect(second.producerPolicyId).not.toBe(first.producerPolicyId);
    expect(second.assetManifestSha256).not.toBe(first.assetManifestSha256);
    expect(second.buildSubjectId).toBe(first.buildSubjectId);
  });
});

async function metadataInput(): Promise<Parameters<
  typeof prepareCppCuteBrowserDistributionMetadata
>[0]> {
  const [buildInputLock, wasmReproducibility, workerBundle, packs, policyBytes] =
    await Promise.all([
      decodeCppCuteBrowserBuildInputLock(
        cppCuteBrowserBuildInputLockResourceBytes(),
      ),
      verifyCppCuteBrowserReproducibilityResource(
        cppCuteBrowserReproducibilityResourceBytes(),
      ),
      verifyCppCuteBrowserWorkerBundle(),
      distributionPacks(),
      cppCuteBrowserProducerTrustPolicyBytes({
        trustStoreSha256: "a".repeat(64),
        builderIds: ["https://builders.browsergrad.dev/distribution-test"],
        keyIds: [`sha256:${"b".repeat(64)}`],
      }),
    ]);
  return {
    buildInputLock,
    packs,
    producerTrustPolicy:
      await admitCppCuteBrowserProducerTrustPolicy(policyBytes),
    wasmReproducibility,
    workerBundle,
  };
}

async function distributionPacks():
Promise<readonly CppCuteBrowserDistributionPackInput[]> {
  return Promise.all(PACK_BINDINGS.map(
    async (binding, index): Promise<CppCuteBrowserDistributionPackInput> => {
      const bytes = await encodeCppCuteBrowserVfsPack([
        {
          virtualPath: `${binding.includeRootId}/fixture-${index}.h`,
          bytes: new TextEncoder().encode(
            `distribution metadata fixture ${binding.includeRootId}\n`,
          ),
        },
      ]);
      return {
        includeRootId:
          binding.includeRootId as CppCuteBrowserDistributionPackIncludeRootId,
        outputPath: binding.outputPath,
        pack: await inspectCppCuteBrowserVfsPack(bytes),
      };
    },
  ));
}
