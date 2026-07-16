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
  unwrapPreparedCppCuteAotJob,
  type PreparedCppCuteAotJob,
} from "./cpp_cute_aot_job.js";
import { unwrapPreparedCppCuteFrontendProfile } from "./cpp_cute_frontend_profile.js";

export const CPP_CUTE_AOT_SANDBOX_POLICY_SCHEMA =
  "browsergrad.compiler.cpp-cute.aot-sandbox-policy";
export const CPP_CUTE_AOT_RESULT_FRAME_PROTOCOL =
  "browsergrad.compiler.cpp-cute.aot-result-frame@1";
export const CPP_CUTE_AOT_RECEIPT_BYTE_LIMIT = 8_388_608;
export const CPP_CUTE_AOT_STDERR_BYTE_LIMIT = 65_536;

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

/**
 * Closed logical policy. Host paths, container IDs, timestamps, and other
 * per-run operational values are intentionally absent from its identity.
 */
export const CPP_CUTE_AOT_SANDBOX_POLICY_V1 = deepFreezeJson({
  schema: CPP_CUTE_AOT_SANDBOX_POLICY_SCHEMA,
  version: { major: 1, minor: 0 },
  contractId: "browsergrad.compiler.cpp-cute.aot@1",
  runtime: {
    engine: "docker",
    endpoint: "local-unix",
    endpointUri: "unix:///var/run/docker.sock",
    platform: "linux/amd64",
    imagePull: "forbidden",
    createBy: "image-config-digest",
    executionEnvironment: "profile-pinned",
  },
  process: {
    user: { uid: 65_532, gid: 65_532 },
    entrypoint: "/opt/browsergrad/bin/cpp-cute-aot-supervisor",
    workingDirectory: "/run/browsergrad",
    arguments: [
      "--profile=/run/browsergrad/control/profile.json",
      "--job=/run/browsergrad/control/job.json",
      "--source-root=/run/browsergrad/source",
      `--protocol=${CPP_CUTE_AOT_RESULT_FRAME_PROTOCOL}`,
    ],
    environment: {
      imageConfig: "must-be-empty",
      overrides: [],
      effective: [],
    },
    stdin: "closed",
    stdout: "single-bounded-frame",
    stderr: "bounded-diagnostic",
    healthcheck: "disabled",
    restart: "none",
    logging: "disabled",
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
    seccomp: "runtime-default-pinned-by-execution-environment",
  },
  filesystem: {
    root: "read-only",
    imageVolumes: "forbidden",
    source: "runner-snapshot-read-only-rprivate",
    control: "runner-snapshot-read-only-rprivate",
    temporaryPath: "/tmp",
    temporary: ["rw", "noexec", "nosuid", "nodev", "mode=1777"],
    temporaryBytes: "min(maxMemoryBytes/4,536870912)",
    dockerSocket: "forbidden",
    devices: "forbidden",
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
      localDockerObservation: "separate-live-authority-required",
      resources: CPP_CUTE_AOT_OCI_RESOURCE_LIMITS,
    },
  },
  decoding: {
    artifact: CPP_CUTE_AOT_ARTIFACT_DECODE_LIMITS,
    receipt: CPP_CUTE_AOT_RECEIPT_DECODE_LIMITS,
    ociManifest: CPP_CUTE_AOT_OCI_MANIFEST_DECODE_LIMITS,
    ociConfig: CPP_CUTE_AOT_OCI_CONFIG_DECODE_LIMITS,
  },
  semantics: {
    linking: "forbidden",
    userProducedNativeExecution: "forbidden",
    externalAttestation: "required-after-receipt",
  },
} as const satisfies JsonObject);

// SHA-256 of canonical JSON for CPP_CUTE_AOT_SANDBOX_POLICY_V1.
export const CPP_CUTE_AOT_SANDBOX_POLICY_SHA256 =
  "c8bf6bdc84ce739ef32939ac8f417433b250ac28218caac95b0eaaeefb7b02ca";

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
): Promise<string> {
  const jobRecord = unwrapPreparedCppCuteAotJob(job);
  const profileRecord = unwrapPreparedCppCuteFrontendProfile(jobRecord.profile);
  const profile = profileRecord.profile;
  if (profile.deployment.sandboxPolicySha256 !== CPP_CUTE_AOT_SANDBOX_POLICY_SHA256) {
    throw new Error("BG-COMPILER-CPP-CUTE-AOT-POLICY-UNSUPPORTED: prepared profile does not name the built-in sandbox policy");
  }
  await verifyCppCuteAotSandboxPolicyIdentity();
  return hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.aot-execution-plan.v1",
    policy: CPP_CUTE_AOT_SANDBOX_POLICY_V1,
    jobId: job.jobId,
    profileHash: job.profileHash,
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
