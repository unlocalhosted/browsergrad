import {
  canonicalJsonBytes,
  hashCanonicalJson,
  sha256Hex,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_SCHEMA,
  computeCppCuteAotExecutionEnvironmentClosureHashes,
  prepareCppCuteAotExecutionEnvironment,
  type CppCuteAotExecutionEnvironmentManifestV1,
  type PreparedCppCuteAotExecutionEnvironment,
} from "../../../src/cpp_cute_aot_environment.js";
import {
  prepareCppCuteFrontendProfile,
  type PreparedCppCuteFrontendProfile,
} from "../../../src/cpp_cute_frontend_profile.js";
import {
  createCppCuteProfileInput,
  type CppCuteProfileFixtureOptions,
} from "./cpp_cute_frontend_fixtures.js";

export interface CppCuteAotExecutionEnvironmentFixture {
  readonly environment: PreparedCppCuteAotExecutionEnvironment;
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly bytes: Uint8Array;
  readonly input: CppCuteAotExecutionEnvironmentManifestV1;
}

export interface CppCuteAotExecutionEnvironmentFixtureOptions {
  readonly profile?: Partial<CppCuteProfileFixtureOptions>;
  readonly mutateBody?: (body: CppCuteAotExecutionEnvironmentManifestV1["body"]) => void;
}

export async function createCppCuteAotExecutionEnvironmentFixture(
  options: CppCuteAotExecutionEnvironmentFixtureOptions = {},
): Promise<CppCuteAotExecutionEnvironmentFixture> {
  const preliminaryProfile = createCppCuteProfileInput(options.profile);
  const body = createEnvironmentBody(preliminaryProfile);
  options.mutateBody?.(body);
  const closureHashes = await computeCppCuteAotExecutionEnvironmentClosureHashes(body);
  asMutableObject(body.image).rootfsManifestSha256 = closureHashes.rootfsManifestSha256;
  asMutableObject(body.toolchain).binariesManifestSha256 = closureHashes.binariesManifestSha256;
  asMutableObject(body.toolchain).dynamicLibrariesManifestSha256 = closureHashes.dynamicLibrariesManifestSha256;
  asMutableObject(body.toolchain).headersManifestSha256 = closureHashes.headersManifestSha256;
  const manifestId = `bg.cpp.execution-environment.sha256.${await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.execution-environment.v1",
    body,
  })}`;
  const input: CppCuteAotExecutionEnvironmentManifestV1 = {
    schema: CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_SCHEMA,
    version: { major: 1, minor: 0 },
    manifestId,
    body,
  };
  const bytes = canonicalJsonBytes(input);
  const profile = await prepareCppCuteFrontendProfile(createCppCuteProfileInput({
    ...options.profile,
    executionEnvironmentManifestSha256: await sha256Hex(bytes),
  }));
  const environment = await prepareCppCuteAotExecutionEnvironment(profile, bytes);
  return { environment, profile, bytes, input };
}

export async function canonicalEnvironmentBytes(
  input: CppCuteAotExecutionEnvironmentManifestV1,
): Promise<Uint8Array> {
  const body = input.body;
  const manifestId = `bg.cpp.execution-environment.sha256.${await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.execution-environment.v1",
    body,
  })}`;
  return canonicalJsonBytes({ ...input, manifestId });
}

function createEnvironmentBody(
  profile: ReturnType<typeof createCppCuteProfileInput>,
): CppCuteAotExecutionEnvironmentManifestV1["body"] {
  const compiler = profile.toolchain.compiler;
  return {
    scope: {
      contractId: "browsergrad.compiler.cpp-cute.aot@1",
      sandboxPolicySha256: profile.deployment.sandboxPolicySha256,
      identity: "environment-only",
      runEvidence: "detached",
      isolation: "single-job-disposable-vm",
    },
    platform: {
      os: "linux",
      architecture: "amd64",
      kernel: {
        release: "6.12.0-browsergrad",
        buildId: "browsergrad-linux-amd64-1",
        imageSha256: "1".repeat(64),
        configSha256: "2".repeat(64),
      },
      runnerIdentity: {
        uid: 65_532,
        gid: 65_532,
        supplementaryGids: [],
        sameUidProcesses: "trusted-boundary",
        coreDumps: "disabled",
        dumpable: false,
      },
      cgroup: {
        version: "v2",
        namespace: "private",
        controllers: ["cpu", "memory", "pids"],
        delegationSha256: "3".repeat(64),
      },
      lsm: [{ kind: "apparmor", policySha256: "4".repeat(64), enforcing: true }],
      clock: {
        monotonic: "CLOCK_MONOTONIC",
        cpuAccounting: "cgroup-v2-cpu-stat-usec",
      },
    },
    runtime: {
      docker: {
        clientVersion: "29.6.1",
        engineVersion: "29.6.1",
        requestApiVersion: "1.49",
        clientBinarySha256: "5".repeat(64),
        daemonConfigSha256: "6".repeat(64),
        imageStore: "containerd",
      },
      containerd: {
        version: "2.1.4",
        binarySha256: "7".repeat(64),
        configSha256: "8".repeat(64),
      },
      runc: {
        version: "1.3.0",
        binarySha256: "9".repeat(64),
      },
      seccomp: {
        mode: "filter",
        profileSha256: "a".repeat(64),
      },
    },
    image: {
      repository: profile.deployment.container.repository,
      platform: "linux/amd64",
      manifestDigest: profile.deployment.container.manifestDigest,
      configDigest: profile.deployment.container.configDigest,
      ociLayoutSha256: "b".repeat(64),
      rootfsManifestSha256: "c".repeat(64),
      buildAttestationSha256: "d".repeat(64),
      layers: [{
        mediaType: "application/vnd.oci.image.layer.v1.tar+gzip",
        digest: `sha256:${"a".repeat(64)}`,
        size: "123" as never,
        diffId: `sha256:${"b".repeat(64)}`,
      }],
    },
    toolchain: {
      binariesManifestSha256: "0".repeat(64),
      dynamicLibrariesManifestSha256: "1".repeat(64),
      headersManifestSha256: "2".repeat(64),
      resourceDirectorySha256: compiler.resourceDirectorySha256,
      binaries: [
        {
          role: "compiler",
          id: compiler.id,
          version: compiler.version,
          buildId: compiler.buildId,
          path: "/opt/browsergrad/llvm/bin/clang++",
          sha256: compiler.binarySha256,
        },
        {
          role: "extractor",
          id: profile.deployment.extractor.id,
          version: profile.deployment.extractor.version,
          buildId: profile.deployment.extractor.buildId,
          path: "/opt/browsergrad/bin/cpp-cute-aot-extractor",
          sha256: profile.deployment.extractor.binarySha256,
        },
        {
          role: "runner",
          id: profile.deployment.runner.id,
          version: profile.deployment.runner.version,
          buildId: null,
          path: "/opt/browsergrad/bin/cpp-cute-aot-supervisor",
          sha256: profile.deployment.runner.binarySha256,
        },
      ],
      dynamicLibraries: [{
        path: "/opt/browsergrad/lib/libclang-cpp.so.20.1",
        sha256: "3".repeat(64),
      }],
      headerSets: profile.toolchain.dependencies.map((dependency) => ({
        dependencyId: dependency.dependencyId,
        kind: dependency.kind,
        version: dependency.version,
        revision: dependency.revision,
        headerSetSha256: dependency.headerSetSha256,
      })),
      includeRoots: profile.virtualFileSystem.includeRoots.map((root) => ({
        includeRootId: root.includeRootId,
        mode: root.mode,
        virtualPath: root.virtualPath,
        manifestSha256: root.manifestSha256,
      })),
    },
    attestation: {
      signer: "external-control-plane",
      evidenceSchema: "browsergrad.compiler.cpp-cute.execution-environment-evidence@1",
      trustStoreSha256: profile.deployment.provenance.trustStoreSha256,
      builderIds: [...profile.deployment.provenance.builderIds],
    },
  } as CppCuteAotExecutionEnvironmentManifestV1["body"];
}

export function cloneEnvironmentInput(
  input: CppCuteAotExecutionEnvironmentManifestV1,
): CppCuteAotExecutionEnvironmentManifestV1 {
  return structuredClone(input) as CppCuteAotExecutionEnvironmentManifestV1;
}

export function asMutableObject(value: unknown): Record<string, unknown> {
  return value as JsonObject as Record<string, unknown>;
}
