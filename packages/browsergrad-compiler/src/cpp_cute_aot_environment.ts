import {
  canonicalJsonBytes,
  decodeWireJson,
  deepFreezeJson,
  encodeWireU64,
  hashCanonicalJson,
  sha256Hex,
  type JsonObject,
  type JsonValue,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  copyInspectedUnsharedUint8Array,
  inspectUnsharedPlainUint8Array,
} from "./cpp_cute_aot_bytes.js";
import {
  unwrapPreparedCppCuteFrontendProfile,
  type CppCuteFrontendDependencyProfile,
  type CppCuteFrontendIncludeRoot,
  type CppCuteFrontendIncludeRootOwner,
  type PreparedCppCuteFrontendProfile,
} from "./cpp_cute_frontend_profile.js";

export const CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_SCHEMA =
  "browsergrad.compiler.cpp-cute.execution-environment";
export const CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_MAJOR = 2;
export const CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_MINOR = 0;
export const CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_EVIDENCE_SCHEMA =
  "browsergrad.compiler.cpp-cute.execution-environment-evidence@1";
export const CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_BYTE_LIMIT = 1_048_576;
export const CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_DECODE_LIMITS = deepFreezeJson({
  maxDocumentBytes: CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_BYTE_LIMIT,
  maxDepth: 24,
  maxNodes: 32_768,
  maxStringBytes: 262_144,
  maxArrayLength: 4_096,
  maxObjectProperties: 1_024,
  maxRank: 64,
  maxIntegerBits: 128,
  maxArithmeticOperations: 65_536,
} as const satisfies JsonObject);

const SHA256_HEX = /^[0-9a-f]{64}$/u;
const OCI_SHA256 = /^sha256:[0-9a-f]{64}$/u;
const MANIFEST_ID = /^bg\.cpp\.execution-environment\.sha256\.[0-9a-f]{64}$/u;
const VERSION = /^[A-Za-z0-9][A-Za-z0-9.+:~_-]{0,127}$/u;
const IDENTIFIER = /^[a-z][a-z0-9._/-]{0,255}$/u;
const DEPENDENCY_ID = /^[a-z][a-z0-9._-]*$/u;
const ABSOLUTE_PATH = /^\/(?:[A-Za-z0-9._+-]+(?:\/[A-Za-z0-9._+-]+)*)?$/u;
const PREPARED_ENVIRONMENTS = new WeakMap<object, StoredCppCuteAotExecutionEnvironmentRecord>();
const ABORT_SIGNAL_ABORTED_GETTER = typeof AbortSignal === "undefined"
  ? undefined
  : Object.getOwnPropertyDescriptor(AbortSignal.prototype, "aborted")?.get;

export interface CppCuteAotExecutionEnvironmentVersion extends JsonObject {
  readonly major: typeof CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_MAJOR;
  readonly minor: typeof CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_MINOR;
}

export interface CppCuteAotExecutionEnvironmentLayer extends JsonObject {
  readonly mediaType:
    | "application/vnd.oci.image.layer.v1.tar"
    | "application/vnd.oci.image.layer.v1.tar+gzip"
    | "application/vnd.oci.image.layer.v1.tar+zstd";
  readonly digest: string;
  readonly size: WireU64;
  readonly diffId: string;
}

export interface CppCuteAotExecutionEnvironmentBinary extends JsonObject {
  readonly role: "runner" | "extractor" | "compiler";
  readonly id: string;
  readonly version: string;
  readonly buildId: string | null;
  readonly path: string;
  readonly sha256: string;
}

export interface CppCuteAotExecutionEnvironmentDynamicLibrary extends JsonObject {
  readonly path: string;
  readonly sha256: string;
}

export interface CppCuteAotExecutionEnvironmentHeaderSet extends JsonObject {
  readonly dependencyId: string;
  readonly kind: CppCuteFrontendDependencyProfile["kind"];
  readonly version: string;
  readonly revision: string;
  readonly headerSetSha256: string;
}

export interface CppCuteAotExecutionEnvironmentIncludeRoot extends JsonObject {
  readonly includeRootId: string;
  readonly mode: CppCuteFrontendIncludeRoot["mode"];
  readonly virtualPath: string;
  readonly manifestSha256: string;
  readonly owner: CppCuteFrontendIncludeRootOwner;
}

export interface CppCuteAotExecutionEnvironmentManifestV2 extends JsonObject {
  readonly schema: typeof CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_SCHEMA;
  readonly version: CppCuteAotExecutionEnvironmentVersion;
  readonly manifestId: string;
  readonly body: JsonObject & {
    readonly scope: JsonObject & {
      readonly contractId: "browsergrad.compiler.cpp-cute.aot@1";
      readonly sandboxPolicySha256: string;
      readonly identity: "environment-only";
      readonly runEvidence: "detached";
      readonly isolation: "single-job-disposable-vm";
    };
    readonly platform: JsonObject & {
      readonly os: "linux";
      readonly architecture: "amd64";
      readonly kernel: JsonObject & {
        readonly release: string;
        readonly buildId: string;
        readonly imageSha256: string;
        readonly configSha256: string;
      };
      readonly runnerIdentity: JsonObject & {
        readonly uid: number;
        readonly gid: number;
        readonly supplementaryGids: readonly number[];
        readonly sameUidProcesses: "trusted-boundary";
        readonly coreDumps: "disabled";
        readonly dumpable: false;
      };
      readonly cgroup: JsonObject & {
        readonly version: "v2";
        readonly namespace: "private";
        readonly controllers: readonly ["cpu", "memory", "pids"];
        readonly delegationSha256: string;
      };
      readonly lsm: readonly (JsonObject & {
        readonly kind: "apparmor" | "selinux";
        readonly policySha256: string;
        readonly enforcing: true;
      })[];
      readonly clock: JsonObject & {
        readonly monotonic: "CLOCK_MONOTONIC";
        readonly cpuAccounting: "cgroup-v2-cpu-stat-usec";
      };
    };
    readonly runtime: JsonObject & {
      readonly docker: JsonObject & {
        readonly clientVersion: string;
        readonly engineVersion: string;
        readonly requestApiVersion: string;
        readonly clientBinarySha256: string;
        readonly daemonConfigSha256: string;
        readonly imageStore: "containerd";
      };
      readonly containerd: JsonObject & {
        readonly version: string;
        readonly binarySha256: string;
        readonly configSha256: string;
      };
      readonly runc: JsonObject & {
        readonly version: string;
        readonly binarySha256: string;
      };
      readonly seccomp: JsonObject & {
        readonly mode: "filter";
        readonly profileSha256: string;
      };
    };
    readonly image: JsonObject & {
      readonly repository: string;
      readonly platform: "linux/amd64";
      readonly manifestDigest: string;
      readonly configDigest: string;
      readonly ociLayoutSha256: string;
      readonly rootfsManifestSha256: string;
      readonly buildAttestationSha256: string;
      readonly layers: readonly CppCuteAotExecutionEnvironmentLayer[];
    };
    readonly toolchain: JsonObject & {
      readonly binariesManifestSha256: string;
      readonly dynamicLibrariesManifestSha256: string;
      readonly headersManifestSha256: string;
      readonly resourceDirectorySha256: string;
      readonly semanticAdapterManifestSha256: string;
      readonly binaries: readonly CppCuteAotExecutionEnvironmentBinary[];
      readonly dynamicLibraries: readonly CppCuteAotExecutionEnvironmentDynamicLibrary[];
      readonly headerSets: readonly CppCuteAotExecutionEnvironmentHeaderSet[];
      readonly includeRoots: readonly CppCuteAotExecutionEnvironmentIncludeRoot[];
    };
    readonly attestation: JsonObject & {
      readonly signer: "external-control-plane";
      readonly evidenceSchema: typeof CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_EVIDENCE_SCHEMA;
      readonly trustStoreSha256: string;
      readonly builderIds: readonly string[];
    };
  };
}

export interface CppCuteAotExecutionEnvironmentClosureHashes {
  readonly rootfsManifestSha256: string;
  readonly binariesManifestSha256: string;
  readonly dynamicLibrariesManifestSha256: string;
  readonly headersManifestSha256: string;
}

declare const preparedCppCuteAotExecutionEnvironmentBrand: unique symbol;

/**
 * Exact canonical environment resource authorized for one prepared profile.
 * This is immutable build/runtime configuration, not evidence that one run
 * actually received the declared kernel, cgroup, mount, or security state.
 */
export interface PreparedCppCuteAotExecutionEnvironment {
  readonly [preparedCppCuteAotExecutionEnvironmentBrand]: true;
  readonly manifestId: string;
  readonly manifestSha256: string;
  readonly manifestByteLength: WireU64;
  readonly bodySha256: string;
  readonly profileHash: string;
}

export interface PreparedCppCuteAotExecutionEnvironmentRecord {
  readonly profile: PreparedCppCuteFrontendProfile;
  readonly manifest: CppCuteAotExecutionEnvironmentManifestV2;
}

interface StoredCppCuteAotExecutionEnvironmentRecord
  extends PreparedCppCuteAotExecutionEnvironmentRecord {
  readonly bytes: Uint8Array;
}

export interface PrepareCppCuteAotExecutionEnvironmentOptions {
  readonly signal?: AbortSignal;
}

export type CppCuteAotExecutionEnvironmentErrorCode =
  | "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-INVALID"
  | "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-UNSUPPORTED-VERSION"
  | "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-HASH-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-PROFILE-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-UNVERIFIED";

export class CppCuteAotExecutionEnvironmentError extends Error {
  constructor(
    readonly code: CppCuteAotExecutionEnvironmentErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteAotExecutionEnvironmentError";
  }
}

/**
 * Snapshots exact bytes before the first await, strict-decodes one canonical
 * manifest, and authorizes its image/toolchain/provenance closure for the
 * exact prepared profile instance.
 */
export async function prepareCppCuteAotExecutionEnvironment(
  profile: PreparedCppCuteFrontendProfile,
  bytes: Uint8Array,
  options: PrepareCppCuteAotExecutionEnvironmentOptions = {},
): Promise<PreparedCppCuteAotExecutionEnvironment> {
  const signal = normalizeOptions(options);
  const snapshot = snapshotBytes(bytes);
  const profileRecord = unwrapPreparedCppCuteFrontendProfile(profile);
  throwIfAborted(signal);
  let value: JsonValue;
  try {
    value = decodeWireJson(snapshot, {
      limits: CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_DECODE_LIMITS,
    });
  } catch (cause) {
    if (cause instanceof CppCuteAotExecutionEnvironmentError) throw cause;
    invalid("$bytes", "execution-environment bytes are not bounded strict JSON", { cause });
  }
  const manifest = parseManifest(value);
  const canonical = canonicalJsonBytes(manifest, {
    limits: CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_DECODE_LIMITS,
  });
  if (!equalBytes(snapshot, canonical)) {
    invalid("$bytes", "execution-environment bytes must use exact canonical JSON encoding");
  }
  throwIfAborted(signal);
  const bodySha256 = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.execution-environment.v2",
    body: manifest.body,
  }, { limits: CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_DECODE_LIMITS });
  const expectedManifestId = `bg.cpp.execution-environment.sha256.${bodySha256}`;
  if (manifest.manifestId !== expectedManifestId) {
    hashMismatch("$.manifestId", `manifest ID must equal ${expectedManifestId}`);
  }
  throwIfAborted(signal);
  await verifyInlineClosureHashes(manifest.body);
  throwIfAborted(signal);
  const manifestSha256 = await sha256Hex(snapshot);
  if (profileRecord.profile.deployment.executionEnvironmentManifestSha256 !== manifestSha256) {
    profileMismatch(
      "$.profile.deployment.executionEnvironmentManifestSha256",
      "prepared profile does not name the exact canonical environment resource",
    );
  }
  verifyProfileClosure(profileRecord.profile, manifest.body);
  throwIfAborted(signal);
  const frozen = deepFreezeJson(manifest) as CppCuteAotExecutionEnvironmentManifestV2;
  const prepared = Object.freeze({
    manifestId: expectedManifestId,
    manifestSha256,
    manifestByteLength: encodeWireU64(BigInt(snapshot.byteLength)),
    bodySha256,
    profileHash: profile.profileHash,
  }) as PreparedCppCuteAotExecutionEnvironment;
  PREPARED_ENVIRONMENTS.set(prepared, Object.freeze({
    profile,
    manifest: frozen,
    bytes: new Uint8Array(snapshot),
  }));
  return prepared;
}

export function unwrapPreparedCppCuteAotExecutionEnvironment(
  environment: PreparedCppCuteAotExecutionEnvironment,
): PreparedCppCuteAotExecutionEnvironmentRecord {
  const record = storedEnvironment(environment);
  return Object.freeze({ profile: record.profile, manifest: record.manifest });
}

/** Disposable canonical bytes for private staging; the copy has no authority. */
export function copyPreparedCppCuteAotExecutionEnvironmentBytes(
  environment: PreparedCppCuteAotExecutionEnvironment,
): Uint8Array {
  return new Uint8Array(storedEnvironment(environment).bytes);
}

/**
 * Derives the identities of closure inventories carried inline by the
 * environment manifest. External OCI-layout/build-attestation resources keep
 * their independent raw-resource hashes.
 */
export async function computeCppCuteAotExecutionEnvironmentClosureHashes(
  body: CppCuteAotExecutionEnvironmentManifestV2["body"],
): Promise<CppCuteAotExecutionEnvironmentClosureHashes> {
  const options = { limits: CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_DECODE_LIMITS } as const;
  const [rootfsManifestSha256, binariesManifestSha256, dynamicLibrariesManifestSha256, headersManifestSha256] =
    await Promise.all([
      hashCanonicalJson({
        domain: "browsergrad.compiler.cpp-cute.execution-environment.rootfs.v2",
        layers: body.image.layers,
      }, options),
      hashCanonicalJson({
        domain: "browsergrad.compiler.cpp-cute.execution-environment.binaries.v2",
        binaries: body.toolchain.binaries,
      }, options),
      hashCanonicalJson({
        domain: "browsergrad.compiler.cpp-cute.execution-environment.dynamic-libraries.v2",
        dynamicLibraries: body.toolchain.dynamicLibraries,
      }, options),
      hashCanonicalJson({
        domain: "browsergrad.compiler.cpp-cute.execution-environment.headers.v2",
        headerSets: body.toolchain.headerSets,
        includeRoots: body.toolchain.includeRoots,
      }, options),
    ]);
  return Object.freeze({
    rootfsManifestSha256,
    binariesManifestSha256,
    dynamicLibrariesManifestSha256,
    headersManifestSha256,
  });
}

async function verifyInlineClosureHashes(
  body: CppCuteAotExecutionEnvironmentManifestV2["body"],
): Promise<void> {
  const expected = await computeCppCuteAotExecutionEnvironmentClosureHashes(body);
  const actual = {
    rootfsManifestSha256: body.image.rootfsManifestSha256,
    binariesManifestSha256: body.toolchain.binariesManifestSha256,
    dynamicLibrariesManifestSha256: body.toolchain.dynamicLibrariesManifestSha256,
    headersManifestSha256: body.toolchain.headersManifestSha256,
  };
  for (const key of Object.keys(expected) as (keyof CppCuteAotExecutionEnvironmentClosureHashes)[]) {
    if (actual[key] !== expected[key]) {
      const path = key === "rootfsManifestSha256"
        ? "$.body.image.rootfsManifestSha256"
        : `$.body.toolchain.${key}`;
      hashMismatch(path, `${key} does not identify its canonical inline closure`);
    }
  }
}

function storedEnvironment(
  environment: PreparedCppCuteAotExecutionEnvironment,
): StoredCppCuteAotExecutionEnvironmentRecord {
  if (typeof environment !== "object" || environment === null) unverified();
  const record = PREPARED_ENVIRONMENTS.get(environment as object);
  if (record === undefined) unverified();
  return record;
}

function parseManifest(value: JsonValue): CppCuteAotExecutionEnvironmentManifestV2 {
  const root = closedObject(value, ["schema", "version", "manifestId", "body"], "$", true);
  if (root.schema !== CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_SCHEMA) {
    invalid("$.schema", `schema must equal ${CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_SCHEMA}`);
  }
  const version = closedObject(field(root, "version", "$"), ["major", "minor"], "$.version", true);
  if (version.major !== CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_MAJOR) {
    unsupported("$.version.major", "execution-environment major version is unsupported");
  }
  if (version.minor !== CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_MINOR) {
    unsupported("$.version.minor", "execution-environment minor version is unsupported");
  }
  const manifestId = boundedPattern(field(root, "manifestId", "$"), "$.manifestId", MANIFEST_ID);
  const body = parseBody(field(root, "body", "$"));
  return {
    schema: CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_SCHEMA,
    version: {
      major: CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_MAJOR,
      minor: CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_MINOR,
    },
    manifestId,
    body,
  };
}

function parseBody(value: JsonValue): CppCuteAotExecutionEnvironmentManifestV2["body"] {
  const body = closedObject(
    value,
    ["scope", "platform", "runtime", "image", "toolchain", "attestation"],
    "$.body",
    true,
  );
  return {
    scope: parseScope(field(body, "scope", "$.body")),
    platform: parsePlatform(field(body, "platform", "$.body")),
    runtime: parseRuntime(field(body, "runtime", "$.body")),
    image: parseImage(field(body, "image", "$.body")),
    toolchain: parseToolchain(field(body, "toolchain", "$.body")),
    attestation: parseAttestation(field(body, "attestation", "$.body")),
  };
}

function parseScope(value: JsonValue): CppCuteAotExecutionEnvironmentManifestV2["body"]["scope"] {
  const path = "$.body.scope";
  const object = closedObject(
    value,
    ["contractId", "sandboxPolicySha256", "identity", "runEvidence", "isolation"],
    path,
    true,
  );
  literal(field(object, "contractId", path), "browsergrad.compiler.cpp-cute.aot@1", `${path}.contractId`);
  literal(field(object, "identity", path), "environment-only", `${path}.identity`);
  literal(field(object, "runEvidence", path), "detached", `${path}.runEvidence`);
  literal(field(object, "isolation", path), "single-job-disposable-vm", `${path}.isolation`);
  return {
    contractId: "browsergrad.compiler.cpp-cute.aot@1",
    sandboxPolicySha256: sha256(field(object, "sandboxPolicySha256", path), `${path}.sandboxPolicySha256`),
    identity: "environment-only",
    runEvidence: "detached",
    isolation: "single-job-disposable-vm",
  };
}

function parsePlatform(value: JsonValue): CppCuteAotExecutionEnvironmentManifestV2["body"]["platform"] {
  const path = "$.body.platform";
  const object = closedObject(
    value,
    ["os", "architecture", "kernel", "runnerIdentity", "cgroup", "lsm", "clock"],
    path,
    true,
  );
  literal(field(object, "os", path), "linux", `${path}.os`);
  literal(field(object, "architecture", path), "amd64", `${path}.architecture`);
  const kernelPath = `${path}.kernel`;
  const kernel = closedObject(
    field(object, "kernel", path),
    ["release", "buildId", "imageSha256", "configSha256"],
    kernelPath,
    true,
  );
  const identityPath = `${path}.runnerIdentity`;
  const identity = closedObject(
    field(object, "runnerIdentity", path),
    ["uid", "gid", "supplementaryGids", "sameUidProcesses", "coreDumps", "dumpable"],
    identityPath,
    true,
  );
  const uid = safeInteger(field(identity, "uid", identityPath), `${identityPath}.uid`, 1, 4_294_967_294);
  const gid = safeInteger(field(identity, "gid", identityPath), `${identityPath}.gid`, 1, 4_294_967_294);
  const supplementary = parseSortedIntegers(
    field(identity, "supplementaryGids", identityPath),
    `${identityPath}.supplementaryGids`,
  );
  literal(field(identity, "sameUidProcesses", identityPath), "trusted-boundary", `${identityPath}.sameUidProcesses`);
  literal(field(identity, "coreDumps", identityPath), "disabled", `${identityPath}.coreDumps`);
  literal(field(identity, "dumpable", identityPath), false, `${identityPath}.dumpable`);
  const cgroupPath = `${path}.cgroup`;
  const cgroup = closedObject(
    field(object, "cgroup", path),
    ["version", "namespace", "controllers", "delegationSha256"],
    cgroupPath,
    true,
  );
  literal(field(cgroup, "version", cgroupPath), "v2", `${cgroupPath}.version`);
  literal(field(cgroup, "namespace", cgroupPath), "private", `${cgroupPath}.namespace`);
  const controllers = stringArray(field(cgroup, "controllers", cgroupPath), `${cgroupPath}.controllers`);
  if (controllers.length !== 3 || controllers[0] !== "cpu" || controllers[1] !== "memory" || controllers[2] !== "pids") {
    invalid(`${cgroupPath}.controllers`, "controllers must be exactly cpu, memory, pids in canonical order");
  }
  const lsm = parseLsm(field(object, "lsm", path), `${path}.lsm`);
  const clockPath = `${path}.clock`;
  const clock = closedObject(
    field(object, "clock", path),
    ["monotonic", "cpuAccounting"],
    clockPath,
    true,
  );
  literal(field(clock, "monotonic", clockPath), "CLOCK_MONOTONIC", `${clockPath}.monotonic`);
  literal(field(clock, "cpuAccounting", clockPath), "cgroup-v2-cpu-stat-usec", `${clockPath}.cpuAccounting`);
  return {
    os: "linux",
    architecture: "amd64",
    kernel: {
      release: boundedVersion(field(kernel, "release", kernelPath), `${kernelPath}.release`),
      buildId: boundedVersion(field(kernel, "buildId", kernelPath), `${kernelPath}.buildId`),
      imageSha256: sha256(field(kernel, "imageSha256", kernelPath), `${kernelPath}.imageSha256`),
      configSha256: sha256(field(kernel, "configSha256", kernelPath), `${kernelPath}.configSha256`),
    },
    runnerIdentity: {
      uid,
      gid,
      supplementaryGids: supplementary,
      sameUidProcesses: "trusted-boundary",
      coreDumps: "disabled",
      dumpable: false,
    },
    cgroup: {
      version: "v2",
      namespace: "private",
      controllers: ["cpu", "memory", "pids"],
      delegationSha256: sha256(field(cgroup, "delegationSha256", cgroupPath), `${cgroupPath}.delegationSha256`),
    },
    lsm,
    clock: {
      monotonic: "CLOCK_MONOTONIC",
      cpuAccounting: "cgroup-v2-cpu-stat-usec",
    },
  };
}

function parseRuntime(value: JsonValue): CppCuteAotExecutionEnvironmentManifestV2["body"]["runtime"] {
  const path = "$.body.runtime";
  const object = closedObject(value, ["docker", "containerd", "runc", "seccomp"], path, true);
  const dockerPath = `${path}.docker`;
  const docker = closedObject(
    field(object, "docker", path),
    ["clientVersion", "engineVersion", "requestApiVersion", "clientBinarySha256", "daemonConfigSha256", "imageStore"],
    dockerPath,
    true,
  );
  literal(field(docker, "imageStore", dockerPath), "containerd", `${dockerPath}.imageStore`);
  const containerdPath = `${path}.containerd`;
  const containerd = closedObject(
    field(object, "containerd", path),
    ["version", "binarySha256", "configSha256"],
    containerdPath,
    true,
  );
  const runcPath = `${path}.runc`;
  const runc = closedObject(field(object, "runc", path), ["version", "binarySha256"], runcPath, true);
  const seccompPath = `${path}.seccomp`;
  const seccomp = closedObject(field(object, "seccomp", path), ["mode", "profileSha256"], seccompPath, true);
  literal(field(seccomp, "mode", seccompPath), "filter", `${seccompPath}.mode`);
  return {
    docker: {
      clientVersion: boundedVersion(field(docker, "clientVersion", dockerPath), `${dockerPath}.clientVersion`),
      engineVersion: boundedVersion(field(docker, "engineVersion", dockerPath), `${dockerPath}.engineVersion`),
      requestApiVersion: boundedVersion(field(docker, "requestApiVersion", dockerPath), `${dockerPath}.requestApiVersion`),
      clientBinarySha256: sha256(field(docker, "clientBinarySha256", dockerPath), `${dockerPath}.clientBinarySha256`),
      daemonConfigSha256: sha256(field(docker, "daemonConfigSha256", dockerPath), `${dockerPath}.daemonConfigSha256`),
      imageStore: "containerd",
    },
    containerd: {
      version: boundedVersion(field(containerd, "version", containerdPath), `${containerdPath}.version`),
      binarySha256: sha256(field(containerd, "binarySha256", containerdPath), `${containerdPath}.binarySha256`),
      configSha256: sha256(field(containerd, "configSha256", containerdPath), `${containerdPath}.configSha256`),
    },
    runc: {
      version: boundedVersion(field(runc, "version", runcPath), `${runcPath}.version`),
      binarySha256: sha256(field(runc, "binarySha256", runcPath), `${runcPath}.binarySha256`),
    },
    seccomp: {
      mode: "filter",
      profileSha256: sha256(field(seccomp, "profileSha256", seccompPath), `${seccompPath}.profileSha256`),
    },
  };
}

function parseImage(value: JsonValue): CppCuteAotExecutionEnvironmentManifestV2["body"]["image"] {
  const path = "$.body.image";
  const object = closedObject(
    value,
    ["repository", "platform", "manifestDigest", "configDigest", "ociLayoutSha256", "rootfsManifestSha256", "buildAttestationSha256", "layers"],
    path,
    true,
  );
  literal(field(object, "platform", path), "linux/amd64", `${path}.platform`);
  const layerValues = array(field(object, "layers", path), `${path}.layers`);
  if (layerValues.length === 0 || layerValues.length > 256) {
    resource(`${path}.layers`, "environment image must contain 1..256 layers");
  }
  const layers = layerValues.map((entry, index): CppCuteAotExecutionEnvironmentLayer => {
    const layerPath = `${path}.layers[${index}]`;
    const layer = closedObject(entry, ["mediaType", "digest", "size", "diffId"], layerPath, true);
    const mediaType = boundedString(field(layer, "mediaType", layerPath), `${layerPath}.mediaType`, 128);
    if (![
      "application/vnd.oci.image.layer.v1.tar",
      "application/vnd.oci.image.layer.v1.tar+gzip",
      "application/vnd.oci.image.layer.v1.tar+zstd",
    ].includes(mediaType)) invalid(`${layerPath}.mediaType`, "unsupported OCI layer media type");
    return {
      mediaType: mediaType as CppCuteAotExecutionEnvironmentLayer["mediaType"],
      digest: ociDigest(field(layer, "digest", layerPath), `${layerPath}.digest`),
      size: wirePositive(field(layer, "size", layerPath), `${layerPath}.size`),
      diffId: ociDigest(field(layer, "diffId", layerPath), `${layerPath}.diffId`),
    };
  });
  return {
    repository: boundedString(field(object, "repository", path), `${path}.repository`, 512),
    platform: "linux/amd64",
    manifestDigest: ociDigest(field(object, "manifestDigest", path), `${path}.manifestDigest`),
    configDigest: ociDigest(field(object, "configDigest", path), `${path}.configDigest`),
    ociLayoutSha256: sha256(field(object, "ociLayoutSha256", path), `${path}.ociLayoutSha256`),
    rootfsManifestSha256: sha256(field(object, "rootfsManifestSha256", path), `${path}.rootfsManifestSha256`),
    buildAttestationSha256: sha256(field(object, "buildAttestationSha256", path), `${path}.buildAttestationSha256`),
    layers,
  };
}

function parseToolchain(value: JsonValue): CppCuteAotExecutionEnvironmentManifestV2["body"]["toolchain"] {
  const path = "$.body.toolchain";
  const object = closedObject(
    value,
    [
      "binariesManifestSha256",
      "dynamicLibrariesManifestSha256",
      "headersManifestSha256",
      "resourceDirectorySha256",
      "semanticAdapterManifestSha256",
      "binaries",
      "dynamicLibraries",
      "headerSets",
      "includeRoots",
    ],
    path,
    true,
  );
  const binaries = parseBinaries(field(object, "binaries", path), `${path}.binaries`);
  const dynamicLibraries = parseDynamicLibraries(
    field(object, "dynamicLibraries", path),
    `${path}.dynamicLibraries`,
  );
  const headerSets = parseHeaderSets(field(object, "headerSets", path), `${path}.headerSets`);
  const includeRoots = parseIncludeRoots(field(object, "includeRoots", path), `${path}.includeRoots`);
  const resourceDirectorySha256 = sha256(
    field(object, "resourceDirectorySha256", path),
    `${path}.resourceDirectorySha256`,
  );
  validateHeaderClosure(headerSets, includeRoots, resourceDirectorySha256);
  return {
    binariesManifestSha256: sha256(field(object, "binariesManifestSha256", path), `${path}.binariesManifestSha256`),
    dynamicLibrariesManifestSha256: sha256(field(object, "dynamicLibrariesManifestSha256", path), `${path}.dynamicLibrariesManifestSha256`),
    headersManifestSha256: sha256(field(object, "headersManifestSha256", path), `${path}.headersManifestSha256`),
    resourceDirectorySha256,
    semanticAdapterManifestSha256: sha256(
      field(object, "semanticAdapterManifestSha256", path),
      `${path}.semanticAdapterManifestSha256`,
    ),
    binaries,
    dynamicLibraries,
    headerSets,
    includeRoots,
  };
}

function parseAttestation(value: JsonValue): CppCuteAotExecutionEnvironmentManifestV2["body"]["attestation"] {
  const path = "$.body.attestation";
  const object = closedObject(
    value,
    ["signer", "evidenceSchema", "trustStoreSha256", "builderIds"],
    path,
    true,
  );
  literal(field(object, "signer", path), "external-control-plane", `${path}.signer`);
  literal(field(object, "evidenceSchema", path), CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_EVIDENCE_SCHEMA, `${path}.evidenceSchema`);
  const builderIds = stringArray(field(object, "builderIds", path), `${path}.builderIds`);
  assertSortedUnique(builderIds, `${path}.builderIds`);
  if (builderIds.length === 0) invalid(`${path}.builderIds`, "at least one environment attestor is required");
  return {
    signer: "external-control-plane",
    evidenceSchema: CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_EVIDENCE_SCHEMA,
    trustStoreSha256: sha256(field(object, "trustStoreSha256", path), `${path}.trustStoreSha256`),
    builderIds,
  };
}

function parseBinaries(value: JsonValue, path: string): readonly CppCuteAotExecutionEnvironmentBinary[] {
  const values = array(value, path);
  if (values.length !== 3) invalid(path, "toolchain requires exactly runner, extractor, and compiler binaries");
  const binaries = values.map((entry, index): CppCuteAotExecutionEnvironmentBinary => {
    const itemPath = `${path}[${index}]`;
    const object = closedObject(entry, ["role", "id", "version", "buildId", "path", "sha256"], itemPath, true);
    const role = boundedString(field(object, "role", itemPath), `${itemPath}.role`, 32);
    if (!(["runner", "extractor", "compiler"] as const).includes(role as never)) {
      invalid(`${itemPath}.role`, "unknown required toolchain binary role");
    }
    const buildId = field(object, "buildId", itemPath);
    if (buildId !== null && typeof buildId !== "string") invalid(`${itemPath}.buildId`, "buildId must be string or null");
    return {
      role: role as CppCuteAotExecutionEnvironmentBinary["role"],
      id: boundedPattern(field(object, "id", itemPath), `${itemPath}.id`, IDENTIFIER),
      version: boundedVersion(field(object, "version", itemPath), `${itemPath}.version`),
      buildId: buildId === null ? null : boundedVersion(buildId, `${itemPath}.buildId`),
      path: absolutePath(field(object, "path", itemPath), `${itemPath}.path`),
      sha256: sha256(field(object, "sha256", itemPath), `${itemPath}.sha256`),
    };
  });
  assertSortedUnique(binaries.map((entry) => entry.role), path);
  if (binaries[0]?.role !== "compiler" || binaries[1]?.role !== "extractor" || binaries[2]?.role !== "runner") {
    invalid(path, "binary roles must be in canonical compiler, extractor, runner order");
  }
  return binaries;
}

function parseDynamicLibraries(
  value: JsonValue,
  path: string,
): readonly CppCuteAotExecutionEnvironmentDynamicLibrary[] {
  const values = array(value, path);
  if (values.length > 1_024) resource(path, "dynamic-library closure exceeds 1024 entries");
  const entries = values.map((entry, index): CppCuteAotExecutionEnvironmentDynamicLibrary => {
    const itemPath = `${path}[${index}]`;
    const object = closedObject(entry, ["path", "sha256"], itemPath, true);
    return {
      path: absolutePath(field(object, "path", itemPath), `${itemPath}.path`),
      sha256: sha256(field(object, "sha256", itemPath), `${itemPath}.sha256`),
    };
  });
  assertSortedUnique(entries.map((entry) => entry.path), path);
  return entries;
}

function parseHeaderSets(value: JsonValue, path: string): readonly CppCuteAotExecutionEnvironmentHeaderSet[] {
  const values = array(value, path);
  if (values.length === 0 || values.length > 128) resource(path, "header-set closure must contain 1..128 entries");
  const entries = values.map((entry, index): CppCuteAotExecutionEnvironmentHeaderSet => {
    const itemPath = `${path}[${index}]`;
    const object = closedObject(
      entry,
      ["dependencyId", "kind", "version", "revision", "headerSetSha256"],
      itemPath,
      true,
    );
    const kind = boundedString(field(object, "kind", itemPath), `${itemPath}.kind`, 64);
    if (!(["cuda-toolkit", "cutlass", "cccl", "cxx-standard-library", "c-system-headers", "linux-sysroot"] as const)
      .includes(kind as never)) {
      invalid(`${itemPath}.kind`, "unknown header-set dependency kind");
    }
    return {
      dependencyId: boundedPattern(field(object, "dependencyId", itemPath), `${itemPath}.dependencyId`, DEPENDENCY_ID),
      kind: kind as CppCuteFrontendDependencyProfile["kind"],
      version: boundedVersion(field(object, "version", itemPath), `${itemPath}.version`),
      revision: boundedVersion(field(object, "revision", itemPath), `${itemPath}.revision`),
      headerSetSha256: sha256(field(object, "headerSetSha256", itemPath), `${itemPath}.headerSetSha256`),
    };
  });
  assertSortedUnique(entries.map((entry) => entry.dependencyId), path);
  return entries;
}

function parseIncludeRoots(value: JsonValue, path: string): readonly CppCuteAotExecutionEnvironmentIncludeRoot[] {
  const values = array(value, path);
  if (values.length === 0 || values.length > 128) resource(path, "include-root closure must contain 1..128 entries");
  const seenIds = new Set<string>();
  const seenPaths = new Set<string>();
  return values.map((entry, index): CppCuteAotExecutionEnvironmentIncludeRoot => {
    const itemPath = `${path}[${index}]`;
    const object = closedObject(
      entry,
      ["includeRootId", "mode", "virtualPath", "manifestSha256", "owner"],
      itemPath,
      true,
    );
    const includeRootId = boundedPattern(
      field(object, "includeRootId", itemPath),
      `${itemPath}.includeRootId`,
      DEPENDENCY_ID,
    );
    if (seenIds.has(includeRootId)) invalid(`${itemPath}.includeRootId`, "include-root IDs must be unique");
    seenIds.add(includeRootId);
    const mode = field(object, "mode", itemPath);
    if (mode !== "quote" && mode !== "system") invalid(`${itemPath}.mode`, "include-root mode must be quote or system");
    const virtualPath = absolutePath(field(object, "virtualPath", itemPath), `${itemPath}.virtualPath`);
    if (seenPaths.has(virtualPath)) invalid(`${itemPath}.virtualPath`, "include-root paths must have unique ownership");
    seenPaths.add(virtualPath);
    return {
      includeRootId,
      mode,
      virtualPath,
      manifestSha256: sha256(field(object, "manifestSha256", itemPath), `${itemPath}.manifestSha256`),
      owner: parseIncludeRootOwner(field(object, "owner", itemPath), `${itemPath}.owner`),
    };
  });
}

function parseIncludeRootOwner(value: JsonValue, path: string): CppCuteFrontendIncludeRootOwner {
  const object = closedObject(value, ["kind", "dependencyId"], path, false);
  const kind = field(object, "kind", path);
  if (kind === "source" || kind === "compiler-resource-directory") {
    closedObject(object, ["kind"], path, true);
    return { kind };
  }
  if (kind === "dependency") {
    const dependency = closedObject(object, ["kind", "dependencyId"], path, true);
    return {
      kind,
      dependencyId: boundedPattern(
        field(dependency, "dependencyId", path),
        `${path}.dependencyId`,
        DEPENDENCY_ID,
      ),
    };
  }
  invalid(`${path}.kind`, "unknown include-root owner kind");
}

function validateHeaderClosure(
  headerSets: readonly CppCuteAotExecutionEnvironmentHeaderSet[],
  includeRoots: readonly CppCuteAotExecutionEnvironmentIncludeRoot[],
  resourceDirectorySha256: string,
): void {
  const dependencies = new Map(headerSets.map((headerSet) => [headerSet.dependencyId, headerSet]));
  const ownedDependencies = new Set<string>();
  let compilerResourceRoots = 0;
  for (const [index, root] of includeRoots.entries()) {
    const path = `$.body.toolchain.includeRoots[${index}]`;
    if (root.owner.kind === "source") continue;
    if (root.owner.kind === "compiler-resource-directory") {
      compilerResourceRoots += 1;
      if (root.manifestSha256 !== resourceDirectorySha256) {
        invalid(`${path}.manifestSha256`, "compiler-owned root differs from the resource-directory manifest");
      }
      continue;
    }
    const dependency = dependencies.get(root.owner.dependencyId);
    if (dependency === undefined) {
      invalid(`${path}.owner.dependencyId`, "include-root owner does not name an environment header set");
    }
    if (root.manifestSha256 !== dependency.headerSetSha256) {
      invalid(`${path}.manifestSha256`, "dependency-owned root differs from its environment header set");
    }
    ownedDependencies.add(dependency.dependencyId);
  }
  if (compilerResourceRoots !== 1) {
    invalid("$.body.toolchain.includeRoots", "environment must expose exactly one compiler resource-directory root");
  }
  for (const headerSet of headerSets) {
    if (!ownedDependencies.has(headerSet.dependencyId)) {
      invalid(
        "$.body.toolchain.includeRoots",
        `header set ${JSON.stringify(headerSet.dependencyId)} must own at least one include root`,
      );
    }
  }
}

function parseLsm(value: JsonValue, path: string): CppCuteAotExecutionEnvironmentManifestV2["body"]["platform"]["lsm"] {
  const values = array(value, path);
  if (values.length === 0 || values.length > 2) invalid(path, "one or two enforcing LSM policies are required");
  const entries = values.map((entry, index): CppCuteAotExecutionEnvironmentManifestV2["body"]["platform"]["lsm"][number] => {
    const itemPath = `${path}[${index}]`;
    const object = closedObject(entry, ["kind", "policySha256", "enforcing"], itemPath, true);
    const kindValue = field(object, "kind", itemPath);
    if (kindValue !== "apparmor" && kindValue !== "selinux") invalid(`${itemPath}.kind`, "LSM kind must be apparmor or selinux");
    const kind: "apparmor" | "selinux" = kindValue;
    literal(field(object, "enforcing", itemPath), true, `${itemPath}.enforcing`);
    return {
      kind,
      policySha256: sha256(field(object, "policySha256", itemPath), `${itemPath}.policySha256`),
      enforcing: true as const,
    };
  });
  assertSortedUnique(entries.map((entry) => entry.kind), path);
  return entries;
}

function verifyProfileClosure(
  profile: ReturnType<typeof unwrapPreparedCppCuteFrontendProfile>["profile"],
  body: CppCuteAotExecutionEnvironmentManifestV2["body"],
): void {
  if (
    body.scope.contractId !== profile.deployment.contractId
    || body.scope.sandboxPolicySha256 !== profile.deployment.sandboxPolicySha256
  ) profileMismatch("$.body.scope", "environment scope differs from prepared profile");
  if (
    body.image.repository !== profile.deployment.container.repository
    || body.image.platform !== profile.deployment.container.platform
    || body.image.manifestDigest !== profile.deployment.container.manifestDigest
    || body.image.configDigest !== profile.deployment.container.configDigest
  ) profileMismatch("$.body.image", "environment image differs from prepared profile");
  const binaries = new Map(body.toolchain.binaries.map((entry) => [entry.role, entry] as const));
  const runner = binaries.get("runner");
  const extractor = binaries.get("extractor");
  const compiler = binaries.get("compiler");
  if (
    runner === undefined
    || runner.id !== profile.deployment.runner.id
    || runner.version !== profile.deployment.runner.version
    || runner.buildId !== null
    || runner.sha256 !== profile.deployment.runner.binarySha256
  ) profileMismatch("$.body.toolchain.binaries.runner", "environment runner differs from prepared profile");
  if (
    extractor === undefined
    || extractor.id !== profile.deployment.extractor.id
    || extractor.version !== profile.deployment.extractor.version
    || extractor.buildId !== profile.deployment.extractor.buildId
    || extractor.sha256 !== profile.deployment.extractor.binarySha256
  ) profileMismatch("$.body.toolchain.binaries.extractor", "environment extractor differs from prepared profile");
  if (
    body.toolchain.semanticAdapterManifestSha256
      !== profile.deployment.extractor.semanticAdapterManifestSha256
  ) {
    profileMismatch(
      "$.body.toolchain.semanticAdapterManifestSha256",
      "environment semantic adapter differs from the prepared extractor profile",
    );
  }
  if (
    compiler === undefined
    || compiler.id !== profile.toolchain.compiler.id
    || compiler.version !== profile.toolchain.compiler.version
    || compiler.buildId !== profile.toolchain.compiler.buildId
    || compiler.sha256 !== profile.toolchain.compiler.binarySha256
    || body.toolchain.resourceDirectorySha256 !== profile.toolchain.compiler.resourceDirectorySha256
  ) profileMismatch("$.body.toolchain.binaries.compiler", "environment compiler differs from prepared profile");
  if (canonicalText(body.toolchain.headerSets) !== canonicalText(profile.toolchain.dependencies)) {
    profileMismatch("$.body.toolchain.headerSets", "environment header sets differ from prepared profile dependencies");
  }
  if (canonicalText(body.toolchain.includeRoots) !== canonicalText(profile.virtualFileSystem.includeRoots)) {
    profileMismatch("$.body.toolchain.includeRoots", "environment include roots differ from prepared profile");
  }
  if (
    body.attestation.trustStoreSha256 !== profile.deployment.provenance.trustStoreSha256
    || canonicalText(body.attestation.builderIds) !== canonicalText(profile.deployment.provenance.builderIds)
  ) profileMismatch("$.body.attestation", "environment attestation policy differs from prepared profile");
}

function snapshotBytes(value: unknown): Uint8Array {
  let inspected;
  try {
    inspected = inspectUnsharedPlainUint8Array(value);
  } catch (cause) {
    invalid("$bytes", "execution-environment input must be an unshared plain Uint8Array", { cause });
  }
  if (inspected.byteLength === 0) invalid("$bytes", "execution-environment bytes must be nonempty");
  if (inspected.byteLength > CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_BYTE_LIMIT) {
    resource("$bytes", `execution-environment bytes exceed ${CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_BYTE_LIMIT}`);
  }
  try {
    return copyInspectedUnsharedUint8Array(value, inspected);
  } catch (cause) {
    invalid("$bytes", "execution-environment bytes became unreadable while snapshotting", { cause });
  }
}

function normalizeOptions(options: PrepareCppCuteAotExecutionEnvironmentOptions): AbortSignal | undefined {
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  try {
    prototype = Object.getPrototypeOf(options);
    descriptors = Object.getOwnPropertyDescriptors(options);
  } catch (cause) {
    invalid("$options", "options must be an inspectable plain object", { cause });
  }
  if (prototype !== Object.prototype) invalid("$options", "options must be a plain object");
  const keys = Reflect.ownKeys(descriptors);
  if (keys.length > 1 || keys.some((key) => key !== "signal")) invalid("$options", "options contain unknown fields");
  const descriptor = descriptors.signal;
  if (descriptor !== undefined && (descriptor.enumerable !== true || !("value" in descriptor))) {
    invalid("$options.signal", "signal must be an enumerable data property");
  }
  const signal = descriptor?.value as unknown;
  if (signal !== undefined && !isAbortSignal(signal)) invalid("$options.signal", "signal must be an AbortSignal");
  return signal;
}

function isAbortSignal(value: unknown): value is AbortSignal {
  if (ABORT_SIGNAL_ABORTED_GETTER === undefined) return false;
  try {
    return typeof ABORT_SIGNAL_ABORTED_GETTER.call(value) === "boolean";
  } catch {
    return false;
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal === undefined) return;
  let aborted: unknown;
  try {
    aborted = ABORT_SIGNAL_ABORTED_GETTER?.call(signal);
  } catch (cause) {
    invalid("$options.signal", "signal is not a readable AbortSignal", { cause });
  }
  if (aborted === true) cancelled();
}

function closedObject(value: JsonValue, keys: readonly string[], path: string, requireAll: boolean): JsonObject {
  if (typeof value !== "object" || value === null || Array.isArray(value)) invalid(path, "expected object");
  const object = value as JsonObject;
  for (const key of Object.keys(object)) if (!keys.includes(key)) invalid(path, `unknown field ${key}`);
  if (requireAll) for (const key of keys) if (!Object.prototype.hasOwnProperty.call(object, key)) invalid(`${path}.${key}`, "required field is missing");
  return object;
}

function field(value: JsonObject, key: string, path: string): JsonValue {
  if (!Object.prototype.hasOwnProperty.call(value, key)) invalid(`${path}.${key}`, "required field is missing");
  return value[key] as JsonValue;
}

function array(value: JsonValue, path: string): readonly JsonValue[] {
  if (!Array.isArray(value)) invalid(path, "expected array");
  return value;
}

function stringArray(value: JsonValue, path: string): readonly string[] {
  return array(value, path).map((entry, index) => boundedString(entry, `${path}[${index}]`, 2_048));
}

function parseSortedIntegers(value: JsonValue, path: string): readonly number[] {
  const result = array(value, path).map((entry, index) => safeInteger(entry, `${path}[${index}]`, 1, 4_294_967_294));
  assertSortedUnique(result, path);
  return result;
}

function assertSortedUnique(values: readonly (string | number)[], path: string): void {
  for (let index = 1; index < values.length; index += 1) {
    const previous = values[index - 1];
    const current = values[index];
    if (previous === undefined || current === undefined || previous >= current) {
      invalid(path, "entries must be strictly sorted and unique");
    }
  }
}

function boundedString(value: JsonValue, path: string, maxBytes: number): string {
  if (typeof value !== "string" || value.length === 0 || new TextEncoder().encode(value).byteLength > maxBytes) {
    invalid(path, "expected bounded nonempty UTF-8 string");
  }
  return value;
}

function boundedPattern(value: JsonValue, path: string, pattern: RegExp): string {
  const text = boundedString(value, path, 2_048);
  if (!pattern.test(text)) invalid(path, "string does not match the required closed format");
  return text;
}

function boundedVersion(value: JsonValue, path: string): string {
  return boundedPattern(value, path, VERSION);
}

function absolutePath(value: JsonValue, path: string): string {
  return boundedPattern(value, path, ABSOLUTE_PATH);
}

function sha256(value: JsonValue, path: string): string {
  return boundedPattern(value, path, SHA256_HEX);
}

function ociDigest(value: JsonValue, path: string): string {
  return boundedPattern(value, path, OCI_SHA256);
}

function wirePositive(value: JsonValue, path: string): WireU64 {
  if (typeof value !== "string" || !/^(?:[1-9][0-9]*)$/u.test(value)) invalid(path, "expected positive canonical u64 string");
  const parsed = BigInt(value);
  if (parsed > 18_446_744_073_709_551_615n) resource(path, "u64 exceeds maximum");
  return value as WireU64;
}

function safeInteger(value: JsonValue, path: string, minimum: number, maximum: number): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < minimum || value > maximum) {
    invalid(path, `expected safe integer in ${minimum}..${maximum}`);
  }
  return value;
}

function literal<T extends string | boolean>(value: JsonValue, expected: T, path: string): asserts value is T {
  if (value !== expected) invalid(path, `must equal ${String(expected)}`);
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  return difference === 0;
}

function canonicalText(value: JsonValue): string {
  return new TextDecoder().decode(canonicalJsonBytes(value, {
    limits: CPP_CUTE_AOT_EXECUTION_ENVIRONMENT_DECODE_LIMITS,
  }));
}

function cancelled(): never {
  throw new CppCuteAotExecutionEnvironmentError(
    "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-CANCELLED",
    "$options.signal",
    "execution-environment preparation was cancelled",
  );
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  throw new CppCuteAotExecutionEnvironmentError(
    "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-INVALID",
    path,
    message,
    options,
  );
}

function unsupported(path: string, message: string): never {
  throw new CppCuteAotExecutionEnvironmentError(
    "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-UNSUPPORTED-VERSION",
    path,
    message,
  );
}

function resource(path: string, message: string): never {
  throw new CppCuteAotExecutionEnvironmentError(
    "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-RESOURCE-LIMIT",
    path,
    message,
  );
}

function hashMismatch(path: string, message: string): never {
  throw new CppCuteAotExecutionEnvironmentError(
    "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-HASH-MISMATCH",
    path,
    message,
  );
}

function profileMismatch(path: string, message: string): never {
  throw new CppCuteAotExecutionEnvironmentError(
    "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-PROFILE-MISMATCH",
    path,
    message,
  );
}

function unverified(): never {
  throw new CppCuteAotExecutionEnvironmentError(
    "BG-COMPILER-CPP-CUTE-AOT-ENVIRONMENT-UNVERIFIED",
    "$",
    "expected an instance-authorized execution environment",
  );
}
