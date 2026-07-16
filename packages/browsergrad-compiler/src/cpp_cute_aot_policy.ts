import {
  deepFreezeJson,
  hashCanonicalJson,
  type JsonObject,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  computeCppCuteAotDependencyManifestHash,
  computeCppCuteAotInvocationManifestHash,
  computeCppCuteAotLimitsManifestHash,
} from "./cpp_cute_aot_manifests.js";
import {
  CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_DECODE_LIMITS,
  unwrapPreparedCppCuteAotExecutionEnvironment,
  type PreparedCppCuteAotExecutionEnvironment,
} from "./cpp_cute_aot_environment.js";
import {
  unwrapPreparedCppCuteAotJob,
  type PreparedCppCuteAotJob,
} from "./cpp_cute_aot_job.js";
import { unwrapPreparedCppCuteFrontendProfile } from "./cpp_cute_frontend_profile.js";

export const CPP_CUTE_AOT_SANDBOX_POLICY_SCHEMA =
  "browsergrad.compiler.cpp-cute.aot-sandbox-policy";
export const CPP_CUTE_AOT_RESULT_FRAME_PROTOCOL =
  "browsergrad.compiler.cpp-cute.aot-result-frame@1";
export const CPP_CUTE_AOT_RESULT_FRAME_HEADER_BYTES = 30;
export const CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_SCHEMA =
  "browsergrad.compiler.cpp-cute.docker-image-inspect";
export const CPP_CUTE_AOT_DOCKER_VERSION_SCHEMA =
  "browsergrad.compiler.cpp-cute.docker-version";
export const CPP_CUTE_AOT_DOCKER_INFO_SCHEMA =
  "browsergrad.compiler.cpp-cute.docker-info";
export const CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_SCHEMA =
  "browsergrad.compiler.cpp-cute.docker-container-inspect";
export const CPP_CUTE_AOT_DOCKER_CONTAINER_RECOVERY_SCHEMA =
  "browsergrad.compiler.cpp-cute.docker-container-recovery";
export const CPP_CUTE_AOT_DOCKER_EXECUTABLE = "/usr/bin/docker";
export const CPP_CUTE_AOT_CONTAINER_HOSTNAME = "browsergrad-cpp-cute-aot";
export const CPP_CUTE_AOT_CONTAINER_SOURCE_ROOT = "/run/browsergrad/source";
export const CPP_CUTE_AOT_CONTAINER_CONTROL_ROOT = "/run/browsergrad/control";
export const CPP_CUTE_AOT_CONTAINER_PROFILE_PATH =
  `${CPP_CUTE_AOT_CONTAINER_CONTROL_ROOT}/profile.json`;
export const CPP_CUTE_AOT_CONTAINER_JOB_PATH =
  `${CPP_CUTE_AOT_CONTAINER_CONTROL_ROOT}/job.json`;
export const CPP_CUTE_AOT_CONTAINER_EXECUTION_ENVIRONMENT_PATH =
  `${CPP_CUTE_AOT_CONTAINER_CONTROL_ROOT}/execution-environment.json`;
export const CPP_CUTE_AOT_DOCKER_DEFAULT_PATH =
  "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
export const CPP_CUTE_AOT_DOCKER_CLIENT_VERSION = "29.6.1";
export const CPP_CUTE_AOT_DOCKER_ENGINE_VERSION = "29.6.1";
export const CPP_CUTE_AOT_DOCKER_API_VERSION = "1.49";
export const CPP_CUTE_AOT_DOCKER_CLIENT_DEFAULT_API_VERSION = "1.55";
export const CPP_CUTE_AOT_DOCKER_ENGINE_API_VERSION = "1.55";
export const CPP_CUTE_AOT_DOCKER_ENGINE_MIN_API_VERSION = "1.40";
export const CPP_CUTE_AOT_DOCKER_VERSION_FORMAT =
  `{"client":{"apiVersion":{{json .Client.APIVersion}},"defaultApiVersion":{{json .Client.DefaultAPIVersion}},"version":{{json .Client.Version}}},"schema":"${CPP_CUTE_AOT_DOCKER_VERSION_SCHEMA}","server":{"apiVersion":{{json .Server.APIVersion}},"arch":{{json .Server.Arch}},"minApiVersion":{{json .Server.MinAPIVersion}},"os":{{json .Server.Os}},"version":{{json .Server.Version}}},"version":1}`;
export const CPP_CUTE_AOT_DOCKER_INFO_FORMAT =
  `{"architecture":{{json .Architecture}},"driverStatus":{{json .DriverStatus}},"osType":{{json .OSType}},"schema":"${CPP_CUTE_AOT_DOCKER_INFO_SCHEMA}","serverVersion":{{json .ServerVersion}},"version":1}`;
export const CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_FORMAT =
  `{"config":{{json .Config}},"descriptor":{"digest":{{json .Descriptor.Digest}},"mediaType":{{json .Descriptor.MediaType}},"size":{{json .Descriptor.Size}}},"id":{{json .Id}},"platform":{"architecture":{{json .Architecture}},"os":{{json .Os}},"osVersion":{{json .OsVersion}},"variant":{{json .Variant}}},"repoDigests":{{json .RepoDigests}},"rootfs":{"diffIds":{{json .RootFS.Layers}},"type":{{json .RootFS.Type}}},"schema":"${CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_SCHEMA}","version":1}`;
export const CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_FORMAT =
  `{"args":{{json .Args}},"config":{{json .Config}},"hostConfig":{{json .HostConfig}},"id":{{json .Id}},"image":{{json .Image}},"imageManifestDescriptor":{{json .ImageManifestDescriptor}},"mounts":{{json .Mounts}},"name":{{json .Name}},"path":{{json .Path}},"restartCount":{{json .RestartCount}},"schema":"${CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_SCHEMA}","state":{{json .State}},"version":1}`;
export const CPP_CUTE_AOT_DOCKER_CONTAINER_RECOVERY_FORMAT =
  `{"id":{{json .Id}},"labels":{{json .Config.Labels}},"name":{{json .Name}},"schema":"${CPP_CUTE_AOT_DOCKER_CONTAINER_RECOVERY_SCHEMA}","version":1}`;
export const CPP_CUTE_AOT_RECEIPT_BYTE_LIMIT = 8_388_608;
export const CPP_CUTE_AOT_STDERR_BYTE_LIMIT = 65_536;
export const CPP_CUTE_AOT_HARD_FRAME_BYTE_LIMIT =
  67_108_864 + CPP_CUTE_AOT_RECEIPT_BYTE_LIMIT + CPP_CUTE_AOT_RESULT_FRAME_HEADER_BYTES;

export const CPP_CUTE_AOT_ARTIFACT_DECODE_LIMITS = deepFreezeJson({
  maxDocumentBytes: 67_108_864,
  maxDepth: 64,
  maxNodes: 100_000,
  maxStringBytes: 2_097_152,
  maxArrayLength: 100_000,
  maxObjectProperties: 10_000,
  maxRank: 64,
  maxIntegerBits: 256,
  maxArithmeticOperations: 200_000,
} as const satisfies JsonObject);

export const CPP_CUTE_AOT_RECEIPT_DECODE_LIMITS = deepFreezeJson({
  maxDocumentBytes: CPP_CUTE_AOT_RECEIPT_BYTE_LIMIT,
  maxDepth: 64,
  maxNodes: 100_000,
  maxStringBytes: 2_097_152,
  maxArrayLength: 100_000,
  maxObjectProperties: 10_000,
  maxRank: 64,
  maxIntegerBits: 256,
  maxArithmeticOperations: 200_000,
} as const satisfies JsonObject);

export const CPP_CUTE_AOT_OCI_RESOURCE_LIMITS = deepFreezeJson({
  manifestBytes: 1_048_576,
  configBytes: 8_388_608,
  layers: 256,
  layerBytes: 2_147_483_648,
  totalLayerBytes: 4_294_967_296,
  historyEntries: 512,
  annotations: 128,
  annotationKeyBytes: 1_024,
  annotationValueBytes: 8_192,
} as const satisfies JsonObject);

export const CPP_CUTE_AOT_OCI_MANIFEST_DECODE_LIMITS = deepFreezeJson({
  maxDocumentBytes: CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.manifestBytes,
  maxDepth: 16,
  maxNodes: 32_768,
  maxStringBytes: 262_144,
  maxArrayLength: 512,
  maxObjectProperties: 512,
  maxRank: 64,
  maxIntegerBits: 256,
  maxArithmeticOperations: 65_536,
} as const satisfies JsonObject);

export const CPP_CUTE_AOT_OCI_CONFIG_DECODE_LIMITS = deepFreezeJson({
  maxDocumentBytes: CPP_CUTE_AOT_OCI_RESOURCE_LIMITS.configBytes,
  maxDepth: 16,
  maxNodes: 16_384,
  maxStringBytes: 1_048_576,
  maxArrayLength: 1_024,
  maxObjectProperties: 512,
  maxRank: 64,
  maxIntegerBits: 256,
  maxArithmeticOperations: 32_768,
} as const satisfies JsonObject);

export const CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_LIMITS = deepFreezeJson({
  timeoutMs: 10_000,
  killGraceMs: 1_000,
  stdoutBytes: 65_536,
  stderrBytes: 65_536,
  repoDigests: 256,
} as const satisfies JsonObject);

export const CPP_CUTE_AOT_DOCKER_VERSION_LIMITS = deepFreezeJson({
  timeoutMs: 10_000,
  killGraceMs: 1_000,
  stdoutBytes: 4_096,
  stderrBytes: 65_536,
} as const satisfies JsonObject);

export const CPP_CUTE_AOT_DOCKER_INFO_LIMITS = deepFreezeJson({
  timeoutMs: 10_000,
  killGraceMs: 1_000,
  stdoutBytes: 4_096,
  stderrBytes: 65_536,
  driverStatusEntries: 1,
} as const satisfies JsonObject);

export const CPP_CUTE_AOT_DOCKER_CREATE_LIMITS = deepFreezeJson({
  timeoutMs: 10_000,
  killGraceMs: 1_000,
  stdoutBytes: 65,
  stderrBytes: CPP_CUTE_AOT_STDERR_BYTE_LIMIT,
  cidFileBytes: 64,
} as const satisfies JsonObject);

export const CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_LIMITS = deepFreezeJson({
  timeoutMs: 10_000,
  killGraceMs: 1_000,
  stdoutBytes: 262_144,
  stderrBytes: CPP_CUTE_AOT_STDERR_BYTE_LIMIT,
} as const satisfies JsonObject);

export const CPP_CUTE_AOT_DOCKER_START_LIMITS = deepFreezeJson({
  killGraceMs: 1_000,
  stdoutBytes: CPP_CUTE_AOT_HARD_FRAME_BYTE_LIMIT,
  stderrBytes: CPP_CUTE_AOT_STDERR_BYTE_LIMIT,
} as const satisfies JsonObject);

export const CPP_CUTE_AOT_DOCKER_REMOVE_LIMITS = deepFreezeJson({
  timeoutMs: 10_000,
  killGraceMs: 1_000,
  stdoutBytes: 65,
  stderrBytes: CPP_CUTE_AOT_STDERR_BYTE_LIMIT,
} as const satisfies JsonObject);

export const CPP_CUTE_AOT_DOCKER_ABSENCE_LIMITS = deepFreezeJson({
  timeoutMs: 10_000,
  killGraceMs: 1_000,
  stdoutBytes: 65,
  stderrBytes: CPP_CUTE_AOT_STDERR_BYTE_LIMIT,
  recoveryAttempts: 3,
  recoveryIntervalMs: 100,
} as const satisfies JsonObject);

export const CPP_CUTE_AOT_DOCKER_LIFECYCLE_LIMITS = deepFreezeJson({
  overheadMs: 60_000,
  cleanupMs: 30_000,
} as const satisfies JsonObject);

export const CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_DECODE_LIMITS = deepFreezeJson({
  maxDocumentBytes: CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_LIMITS.stdoutBytes,
  maxDepth: 32,
  maxNodes: 32_768,
  maxStringBytes: 1_048_576,
  maxArrayLength: 1_024,
  maxObjectProperties: 1_024,
  maxRank: 64,
  maxIntegerBits: 256,
  maxArithmeticOperations: 65_536,
} as const satisfies JsonObject);

export const CPP_CUTE_AOT_DOCKER_VERSION_DECODE_LIMITS = deepFreezeJson({
  maxDocumentBytes: CPP_CUTE_AOT_DOCKER_VERSION_LIMITS.stdoutBytes,
  maxDepth: 8,
  maxNodes: 64,
  maxStringBytes: 256,
  maxArrayLength: 8,
  maxObjectProperties: 16,
  maxRank: 8,
  maxIntegerBits: 64,
  maxArithmeticOperations: 128,
} as const satisfies JsonObject);

export const CPP_CUTE_AOT_DOCKER_INFO_DECODE_LIMITS = deepFreezeJson({
  maxDocumentBytes: CPP_CUTE_AOT_DOCKER_INFO_LIMITS.stdoutBytes,
  maxDepth: 8,
  maxNodes: 64,
  maxStringBytes: 256,
  maxArrayLength: 8,
  maxObjectProperties: 16,
  maxRank: 8,
  maxIntegerBits: 64,
  maxArithmeticOperations: 128,
} as const satisfies JsonObject);

export const CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_DECODE_LIMITS = deepFreezeJson({
  maxDocumentBytes: CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_LIMITS.stdoutBytes,
  maxDepth: 32,
  maxNodes: 65_536,
  maxStringBytes: 2_097_152,
  maxArrayLength: 4_096,
  maxObjectProperties: 4_096,
  maxRank: 64,
  maxIntegerBits: 256,
  maxArithmeticOperations: 131_072,
} as const satisfies JsonObject);

/**
 * Closed logical policy. Per-run staging paths, container IDs, timestamps,
 * and other random operational values are intentionally absent. The fixed
 * Docker executable path is part of policy identity.
 */
export const CPP_CUTE_AOT_SANDBOX_POLICY_V1 = deepFreezeJson({
  schema: CPP_CUTE_AOT_SANDBOX_POLICY_SCHEMA,
  version: { major: 1, minor: 2 },
  contractId: "browsergrad.compiler.cpp-cute.aot@1",
  runtime: {
    engine: "docker",
    executable: CPP_CUTE_AOT_DOCKER_EXECUTABLE,
    clientVersion: CPP_CUTE_AOT_DOCKER_CLIENT_VERSION,
    engineVersion: CPP_CUTE_AOT_DOCKER_ENGINE_VERSION,
    requestApiVersion: CPP_CUTE_AOT_DOCKER_API_VERSION,
    clientDefaultApiVersion: CPP_CUTE_AOT_DOCKER_CLIENT_DEFAULT_API_VERSION,
    engineApiVersion: CPP_CUTE_AOT_DOCKER_ENGINE_API_VERSION,
    engineMinApiVersion: CPP_CUTE_AOT_DOCKER_ENGINE_MIN_API_VERSION,
    imageStore: "containerd",
    endpoint: "local-unix",
    endpointUri: "unix:///var/run/docker.sock",
    platform: "linux/amd64",
    imagePull: "forbidden",
    createBy: "authorized-manifest-reference",
    executionEnvironment: "profile-hash-pinned-external-attestation-required",
    client: {
      config: "runner-private-empty-directory",
      cwd: "runner-private-directory",
      inheritedEnvironment: "forbidden",
      environment: [
        `DOCKER_API_VERSION=${CPP_CUTE_AOT_DOCKER_API_VERSION}`,
        "HOME=runner-private-directory",
        "LANG=C",
        "LC_ALL=C",
        "TZ=UTC",
      ],
      shell: false,
      retries: 0,
    },
    hostProcess: {
      isolation: "dedicated-single-job-worker-required",
      uid: "dedicated-runner-required",
      sameUidProcesses: "trusted-boundary",
      coreDumps: "disabled-required",
      failStop: "abort-on-unreaped-child-or-unproved-container-absence",
      authority: "execution-environment-attestation-required",
    },
  },
  process: {
    user: { uid: 65_532, gid: 65_532 },
    hostname: CPP_CUTE_AOT_CONTAINER_HOSTNAME,
    entrypoint: "/opt/browsergrad/bin/cpp-cute-aot-supervisor",
    workingDirectory: "/run/browsergrad",
    arguments: [
      "--profile=/run/browsergrad/control/profile.json",
      "--job=/run/browsergrad/control/job.json",
      "--execution-environment=/run/browsergrad/control/execution-environment.json",
      "--source-root=/run/browsergrad/source",
      `--protocol=${CPP_CUTE_AOT_RESULT_FRAME_PROTOCOL}`,
    ],
    environment: {
      imageConfig: "must-be-empty",
      overrides: [],
      dockerInjected: [
        `PATH=${CPP_CUTE_AOT_DOCKER_DEFAULT_PATH}`,
        `HOSTNAME=${CPP_CUTE_AOT_CONTAINER_HOSTNAME}`,
      ],
      supervisor: "clear-before-toolchain",
      effective: [],
    },
    stdin: "closed",
    stdout: "single-bounded-frame",
    stderr: "bounded-diagnostic",
    healthcheck: "disabled",
    restart: "none",
    logging: "disabled",
    runtime: "runc",
  },
  namespaces: {
    network: "none",
    ipc: "none",
    pid: "private",
    uts: "private",
    cgroup: "private",
  },
  privileges: {
    privileged: false,
    capabilitiesAdded: [],
    capabilitiesDropped: ["ALL"],
    noNewPrivileges: true,
    seccomp: "runtime-default-requested-effective-profile-requires-external-attestation",
  },
  filesystem: {
    root: "image-rootfs-read-only",
    runtimeManagedMounts: [
      "/dev",
      "/etc/hostname",
      "/etc/hosts",
      "/etc/resolv.conf",
      "/proc",
      "/sys",
    ],
    imageVolumes: "forbidden",
    source: "runner-snapshot-read-only-rprivate",
    control: "runner-snapshot-read-only-rprivate",
    temporaryPath: "/tmp",
    temporary: ["rw", "noexec", "nosuid", "nodev", "mode=1777"],
    temporaryBytes: "max(1,min(floor(maxMemoryBytes/4),536870912))",
    dockerSocket: "forbidden",
    devices: "runtime-minimal-defaults",
    hostDevicePassthrough: "forbidden",
  },
  limits: {
    memory: "maxMemoryBytes",
    memorySwap: "maxMemoryBytes",
    processes: "maxProcesses",
    wallTime: "maxWallTimeMs",
    cpuTime: "in-image-supervisor-and-cgroup-observation",
    stdout: "maxOutputBytes-plus-receipt-cap",
    stderrBytes: CPP_CUTE_AOT_STDERR_BYTE_LIMIT,
    receiptBytes: CPP_CUTE_AOT_RECEIPT_BYTE_LIMIT,
  },
  evidence: {
    ociMetadata: {
      authority: "manifest-config-metadata-only",
      layerBlobs: "not-supplied-or-rehashed",
      localDockerObservation: {
        authority: "shell-owned-point-in-time",
        sequence: ["version", "info", "image-inspect"],
        runtime: {
          version: {
            schema: `${CPP_CUTE_AOT_DOCKER_VERSION_SCHEMA}@1`,
            format: CPP_CUTE_AOT_DOCKER_VERSION_FORMAT,
            limits: CPP_CUTE_AOT_DOCKER_VERSION_LIMITS,
          },
          info: {
            schema: `${CPP_CUTE_AOT_DOCKER_INFO_SCHEMA}@1`,
            format: CPP_CUTE_AOT_DOCKER_INFO_FORMAT,
            limits: CPP_CUTE_AOT_DOCKER_INFO_LIMITS,
          },
          imageStoreEvidence: ["exact-driver-status", "platform-selected-manifest-id"],
        },
        image: "authorized-manifest-reference",
        platform: "linux/amd64",
        projectionSchema: `${CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_SCHEMA}@1`,
        projectionFormat: CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_FORMAT,
        config: "recursively-semantically-empty",
        descriptor: "exact-authorized-manifest",
        rootfs: "exact-ordered-authorized-diff-ids",
        continuedPresence: "not-claimed",
        limits: CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_LIMITS,
      },
      containerLifecycle: {
        sequence: [
          "create",
          "inspect-created",
          "start-attached",
          "inspect-terminal",
          "remove",
          "prove-absent",
        ],
        createBy: "authorized-manifest-reference",
        hostname: CPP_CUTE_AOT_CONTAINER_HOSTNAME,
        output: "attach-before-start-single-frame",
        cleanup: "exact-container-id-before-staging-removal",
        createdProjectionSchema: `${CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_SCHEMA}@1`,
        createdProjectionFormat: CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_FORMAT,
        createdProjectionAuthority: "daemon-recorded-request-and-state-not-kernel-enforcement",
        recoveryProjectionSchema: `${CPP_CUTE_AOT_DOCKER_CONTAINER_RECOVERY_SCHEMA}@1`,
        recoveryProjectionFormat: CPP_CUTE_AOT_DOCKER_CONTAINER_RECOVERY_FORMAT,
        limits: {
          create: CPP_CUTE_AOT_DOCKER_CREATE_LIMITS,
          inspect: CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_LIMITS,
          start: CPP_CUTE_AOT_DOCKER_START_LIMITS,
          remove: CPP_CUTE_AOT_DOCKER_REMOVE_LIMITS,
          absence: CPP_CUTE_AOT_DOCKER_ABSENCE_LIMITS,
          lifecycle: CPP_CUTE_AOT_DOCKER_LIFECYCLE_LIMITS,
        },
      },
      resources: CPP_CUTE_AOT_OCI_RESOURCE_LIMITS,
    },
  },
  decoding: {
    artifact: CPP_CUTE_AOT_ARTIFACT_DECODE_LIMITS,
    receipt: CPP_CUTE_AOT_RECEIPT_DECODE_LIMITS,
    executionEnvironment: CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_DECODE_LIMITS,
    ociManifest: CPP_CUTE_AOT_OCI_MANIFEST_DECODE_LIMITS,
    ociConfig: CPP_CUTE_AOT_OCI_CONFIG_DECODE_LIMITS,
    dockerVersion: CPP_CUTE_AOT_DOCKER_VERSION_DECODE_LIMITS,
    dockerInfo: CPP_CUTE_AOT_DOCKER_INFO_DECODE_LIMITS,
    dockerImageInspect: CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_DECODE_LIMITS,
    dockerContainerInspect: CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_DECODE_LIMITS,
  },
  semantics: {
    linking: "forbidden",
    userProducedNativeExecution: "forbidden",
    externalAttestation: "required-after-receipt",
  },
} as const satisfies JsonObject);

// SHA-256 of canonical JSON for CPP_CUTE_AOT_SANDBOX_POLICY_V1.
export const CPP_CUTE_AOT_SANDBOX_POLICY_SHA256 =
  "d626d88e137cae80d3df7ccade30b5317f1ecc1906e3604133c5dab32dd278e0";

export async function verifyCppCuteAotSandboxPolicyIdentity(): Promise<void> {
  const actual = await hashCanonicalJson(CPP_CUTE_AOT_SANDBOX_POLICY_V1);
  if (actual !== CPP_CUTE_AOT_SANDBOX_POLICY_SHA256) {
    throw new Error("BG-COMPILER-CPP-CUTE-AOT-POLICY-HASH-MISMATCH: built-in sandbox policy identity drifted");
  }
}

/**
 * Hashes the exact output-independent logical execution plan. Random host
 * workspace paths and container IDs are excluded; every semantic Docker
 * setting is derived from this built-in policy plus the prepared job/profile.
 */
export async function computeCppCuteAotExecutionPlanHash(
  job: PreparedCppCuteAotJob,
  environment: PreparedCppCuteAotExecutionEnvironment,
): Promise<string> {
  const jobRecord = unwrapPreparedCppCuteAotJob(job);
  const profileRecord = unwrapPreparedCppCuteFrontendProfile(jobRecord.profile);
  const environmentRecord = unwrapPreparedCppCuteAotExecutionEnvironment(environment);
  if (environmentRecord.profile !== jobRecord.profile || environment.profileHash !== job.profileHash) {
    throw new Error("BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-PROFILE-MISMATCH: execution environment belongs to a different prepared profile");
  }
  const profile = profileRecord.profile;
  if (profile.deployment.sandboxPolicySha256 !== CPP_CUTE_AOT_SANDBOX_POLICY_SHA256) {
    throw new Error("BG-COMPILER-CPP-CUTE-AOT-POLICY-UNSUPPORTED: prepared profile does not name the built-in sandbox policy");
  }
  await verifyCppCuteAotSandboxPolicyIdentity();
  verifyExecutionEnvironmentPolicy(environmentRecord.manifest);
  return hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.aot-execution-plan.v2",
    policy: CPP_CUTE_AOT_SANDBOX_POLICY_V1,
    jobId: job.jobId,
    profileHash: job.profileHash,
    executionEnvironment: {
      manifestId: environment.manifestId,
      manifestSha256: environment.manifestSha256,
      manifestByteLength: environment.manifestByteLength,
      bodySha256: environment.bodySha256,
    },
    invocationManifestSha256: await computeCppCuteAotInvocationManifestHash(job),
    dependencyManifestSha256: await computeCppCuteAotDependencyManifestHash(jobRecord.profile),
    limitsManifestSha256: await computeCppCuteAotLimitsManifestHash(profile.extractionLimits),
    deployment: profile.deployment,
    toolchain: profile.toolchain,
    language: profile.language,
    target: profile.target,
    virtualFileSystem: profile.virtualFileSystem,
    sourceFiles: jobRecord.job.files,
  });
}

function verifyExecutionEnvironmentPolicy(
  manifest: ReturnType<typeof unwrapPreparedCppCuteAotExecutionEnvironment>["manifest"],
): void {
  const body = manifest.body;
  if (
    body.scope.sandboxPolicySha256 !== CPP_CUTE_AOT_SANDBOX_POLICY_SHA256
    || body.runtime.docker.clientVersion !== CPP_CUTE_AOT_DOCKER_CLIENT_VERSION
    || body.runtime.docker.engineVersion !== CPP_CUTE_AOT_DOCKER_ENGINE_VERSION
    || body.runtime.docker.requestApiVersion !== CPP_CUTE_AOT_DOCKER_API_VERSION
    || body.runtime.docker.imageStore !== "containerd"
    || body.image.platform !== "linux/amd64"
    || body.platform.runnerIdentity.uid !== CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.user.uid
    || body.platform.runnerIdentity.gid !== CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.user.gid
  ) {
    throw new Error("BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-POLICY-MISMATCH: execution environment differs from the built-in sandbox policy");
  }
}
