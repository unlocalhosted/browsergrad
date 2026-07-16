import {
  chmodSync,
  existsSync,
  lstatSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";

import { describe, expect, it, vi } from "vitest";

import {
  authorizeCppCuteAotOciMetadata,
  verifyCppCuteAotOciMetadata,
  type AuthorizedCppCuteAotOciMetadata,
  type VerifiedCppCuteAotOciMetadata,
} from "../dist/cpp_cute_aot_oci.js";
import {
  prepareCppCuteAotExecutionEnvironment as prepareDistEnvironment,
} from "../dist/cpp_cute_aot_environment.js";
import { prepareCppCuteAotJob as prepareDistJob } from "../dist/cpp_cute_aot_job.js";
import {
  copyCppCuteAotOfflineRunStagingInputs,
  encodeCppCuteAotResultFrame,
  prepareCppCuteAotOfflineRun as prepareDistOfflineRun,
  type PreparedCppCuteAotOfflineRun,
  unwrapPreparedCppCuteAotOfflineRun as unwrapDistOfflineRun,
} from "../dist/cpp_cute_aot_runner_plan.js";
import {
  CPP_CUTE_AOT_CONTAINER_CONTROL_ROOT,
  CPP_CUTE_AOT_CONTAINER_HOSTNAME,
  CPP_CUTE_AOT_CONTAINER_SOURCE_ROOT,
  CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_SCHEMA,
  CPP_CUTE_AOT_DOCKER_CONTAINER_RECOVERY_SCHEMA,
  CPP_CUTE_AOT_DOCKER_INFO_SCHEMA,
  CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_SCHEMA,
  CPP_CUTE_AOT_DOCKER_VERSION_SCHEMA,
  CPP_CUTE_AOT_SANDBOX_POLICY_V1,
} from "../dist/cpp_cute_aot_policy.js";
import {
  prepareCppCuteFrontendProfile as prepareDistProfile,
  unwrapPreparedCppCuteFrontendProfile as unwrapDistProfile,
} from "../dist/cpp_cute_frontend_profile.js";
import { unwrapPreparedCppCuteAotJob } from "../src/cpp_cute_aot_job.js";
import {
  copyPreparedCppCuteAotExecutionEnvironmentBytes,
} from "../src/cpp_cute_aot_environment.js";
import {
  copyCppCuteAotOfflineRunSourceBlobs,
  unwrapPreparedCppCuteAotOfflineRun,
} from "../src/cpp_cute_aot_runner_plan.js";
import { unwrapPreparedCppCuteFrontendProfile } from "../src/cpp_cute_frontend_profile.js";
import {
  createCppCuteAotOciFixture,
  type CppCuteAotOciFixture,
} from "../tests/compiler/support/cpp_cute_aot_oci_fixtures.js";
import {
  buildCppCuteAotDockerAbsenceRequest,
  buildCppCuteAotDockerContainerInspectRequest,
  buildCppCuteAotDockerCreateRequest,
  buildCppCuteAotDockerImageInspectRequest,
  buildCppCuteAotDockerInfoRequest,
  buildCppCuteAotDockerRemoveRequest,
  buildCppCuteAotDockerStartAttachedRequest,
  buildCppCuteAotDockerVersionRequest,
  type BoundedChildProcessRequest,
  type BoundedChildProcessResult,
} from "./cpp_cute_aot_docker_process.mjs";
import {
  __executeCppCuteAotDockerRunWithProcessForTest,
  __unwrapCompletedCppCuteAotDockerRunForTest,
  unwrapCompletedCppCuteAotDockerRun,
} from "./cpp_cute_aot_docker_lifecycle.mjs";

const encoder = new TextEncoder();
const decoder = new TextDecoder();
const CONTAINER_ID = "1".repeat(64);
const OTHER_CONTAINER_ID = "2".repeat(64);

type RequestKind =
  | "version"
  | "info"
  | "image"
  | "create"
  | "recovery"
  | "inspect-created"
  | "start"
  | "inspect-terminal"
  | "remove"
  | "absence";

interface PreparedLifecycleFixture {
  readonly fixture: CppCuteAotOciFixture;
  readonly metadata: VerifiedCppCuteAotOciMetadata;
  readonly authorized: AuthorizedCppCuteAotOciMetadata;
  readonly plan: PreparedCppCuteAotOfflineRun;
  readonly frame: Uint8Array;
}

interface ContainerSessionState {
  runRoot?: string;
  configDirectory?: string;
  homeDirectory?: string;
  containerName?: string;
  sessionNonce?: string;
  sourceDirectory?: string;
  controlDirectory?: string;
  cidFile?: string;
  memoryBytes?: number;
  maxProcesses?: number;
  jobId?: string;
  executionPlanSha256?: string;
  removeForced?: boolean;
}

interface StagingSnapshot {
  readonly modes: Readonly<Record<string, number>>;
  readonly bytes: Readonly<Record<string, Uint8Array>>;
}

interface RecordedRequest {
  readonly kind: RequestKind;
  readonly request: BoundedChildProcessRequest;
}

interface AdapterOptions {
  readonly containerId?: string;
  readonly createStdoutId?: string;
  readonly cidFileId?: string | null;
  readonly recoveryBytes?: Uint8Array;
  readonly mutateRecovery?: (projection: Record<string, unknown>) => void;
  readonly mutateCreated?: (projection: Record<string, unknown>) => void;
  readonly mutateTerminal?: (projection: Record<string, unknown>) => void;
  readonly startResult?: BoundedChildProcessResult;
  readonly frameBytes?: Uint8Array;
  readonly removeResult?: BoundedChildProcessResult;
  readonly absenceResult?: BoundedChildProcessResult;
}

async function prepareFixture(): Promise<PreparedLifecycleFixture> {
  const fixture = await createCppCuteAotOciFixture();
  const sourcePlan = unwrapPreparedCppCuteAotOfflineRun(fixture.plan);
  const sourceJob = unwrapPreparedCppCuteAotJob(sourcePlan.job);
  const sourceProfile = unwrapPreparedCppCuteFrontendProfile(sourceJob.profile);
  const distProfile = await prepareDistProfile(structuredClone(sourceProfile.profile));
  const distJob = await prepareDistJob(distProfile, structuredClone(sourceJob.job));
  const distEnvironment = await prepareDistEnvironment(
    distProfile,
    copyPreparedCppCuteAotExecutionEnvironmentBytes(sourcePlan.executionEnvironment),
  );
  const distPlan = await prepareDistOfflineRun(
    distJob,
    distEnvironment,
    copyCppCuteAotOfflineRunSourceBlobs(fixture.plan).map(({ fileId, bytes }) => ({
      fileId,
      bytes: new Uint8Array(bytes),
    })),
  );
  const metadata = await verifyCppCuteAotOciMetadata(fixture.evidence);
  return {
    fixture,
    metadata,
    authorized: authorizeCppCuteAotOciMetadata(distPlan, metadata),
    plan: distPlan,
    frame: encodeCppCuteAotResultFrame(
      fixture.runner.artifactBytes,
      fixture.runner.receiptBytes,
    ),
  };
}

function processResult(
  stdout: Uint8Array | string = new Uint8Array(),
  overrides: Partial<BoundedChildProcessResult> = {},
): BoundedChildProcessResult {
  return Object.freeze({
    exitCode: 0,
    signal: null,
    stdout: typeof stdout === "string" ? encoder.encode(stdout) : new Uint8Array(stdout),
    stderr: new Uint8Array(),
    ...overrides,
  });
}

function jsonResult(value: unknown): BoundedChildProcessResult {
  return processResult(`${JSON.stringify(value)}\n`);
}

function versionProjection(): Record<string, unknown> {
  return {
    client: { apiVersion: "1.49", defaultApiVersion: "1.55", version: "29.6.1" },
    schema: CPP_CUTE_AOT_DOCKER_VERSION_SCHEMA,
    server: {
      apiVersion: "1.55",
      arch: "amd64",
      minApiVersion: "1.40",
      os: "linux",
      version: "29.6.1",
    },
    version: 1,
  };
}

function infoProjection(): Record<string, unknown> {
  return {
    architecture: "x86_64",
    driverStatus: [["driver-type", "io.containerd.snapshotter.v1"]],
    osType: "linux",
    schema: CPP_CUTE_AOT_DOCKER_INFO_SCHEMA,
    serverVersion: "29.6.1",
    version: 1,
  };
}

function imageProjection(prepared: PreparedLifecycleFixture): Record<string, unknown> {
  const config = JSON.parse(decoder.decode(prepared.fixture.evidence.configBytes)) as {
    rootfs: { diff_ids: string[] };
  };
  return {
    config: {
      ArgsEscaped: false,
      AttachStdin: false,
      Cmd: null,
      Entrypoint: [],
      Env: null,
      Labels: {},
      StopTimeout: 0,
      User: "",
      WorkingDir: "",
    },
    descriptor: {
      digest: prepared.fixture.manifestDigest,
      mediaType: prepared.metadata.manifest.mediaType,
      size: prepared.metadata.manifest.size,
    },
    id: prepared.fixture.manifestDigest,
    platform: { architecture: "amd64", os: "linux", osVersion: "", variant: "" },
    repoDigests: [prepared.authorized.imageReference],
    rootfs: { diffIds: config.rootfs.diff_ids, type: "layers" },
    schema: CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_SCHEMA,
    version: 1,
  };
}

function required<T>(value: T | undefined, name: string): T {
  if (value === undefined) throw new Error(`adapter state missing ${name}`);
  return value;
}

function argumentValue(arguments_: readonly string[], prefix: string): string {
  const value = arguments_.find((entry) => entry.startsWith(prefix));
  if (value === undefined) throw new Error(`request missing ${prefix}`);
  return value.slice(prefix.length);
}

function mountSource(arguments_: readonly string[], target: string): string {
  for (let index = 0; index < arguments_.length; index += 1) {
    if (arguments_[index] !== "--mount") continue;
    const specification = arguments_[index + 1];
    if (specification?.includes(`target=${target}`)) {
      const source = specification.split(",").find((entry) => entry.startsWith("source="));
      if (source !== undefined) return source.slice("source=".length);
    }
  }
  throw new Error(`request missing mount ${target}`);
}

function captureCreateState(
  request: BoundedChildProcessRequest,
  state: ContainerSessionState,
): void {
  state.runRoot = request.cwd;
  state.configDirectory = argumentAfter(request.arguments, "--config");
  state.homeDirectory = required(request.environment.HOME, "HOME");
  state.containerName = argumentValue(request.arguments, "--name=");
  state.sessionNonce = argumentValue(request.arguments, "--label=browsergrad.session=");
  state.jobId = argumentValue(request.arguments, "--label=browsergrad.job=");
  state.executionPlanSha256 = argumentValue(request.arguments, "--label=browsergrad.plan=");
  state.cidFile = argumentValue(request.arguments, "--cidfile=");
  state.sourceDirectory = mountSource(request.arguments, CPP_CUTE_AOT_CONTAINER_SOURCE_ROOT);
  state.controlDirectory = mountSource(request.arguments, CPP_CUTE_AOT_CONTAINER_CONTROL_ROOT);
  state.memoryBytes = Number(argumentValue(request.arguments, "--memory="));
  state.maxProcesses = Number(argumentValue(request.arguments, "--pids-limit="));
}

function argumentAfter(arguments_: readonly string[], argument: string): string {
  const index = arguments_.indexOf(argument);
  const value = arguments_[index + 1];
  if (index < 0 || value === undefined) throw new Error(`request missing ${argument}`);
  return value;
}

function containerLabels(state: ContainerSessionState): Record<string, string> {
  return {
    "browsergrad.job": required(state.jobId, "jobId"),
    "browsergrad.owner": "cpp-cute-aot",
    "browsergrad.plan": required(state.executionPlanSha256, "executionPlanSha256"),
    "browsergrad.session": required(state.sessionNonce, "sessionNonce"),
  };
}

function recoveryProjection(
  state: ContainerSessionState,
  containerId: string,
): Record<string, unknown> {
  return {
    id: containerId,
    labels: containerLabels(state),
    name: `/${required(state.containerName, "containerName")}`,
    schema: CPP_CUTE_AOT_DOCKER_CONTAINER_RECOVERY_SCHEMA,
    version: 1,
  };
}

function containerProjection(
  prepared: PreparedLifecycleFixture,
  state: ContainerSessionState,
  containerId: string,
  status: "created" | "exited",
): Record<string, unknown> {
  const sourceDirectory = required(state.sourceDirectory, "sourceDirectory");
  const controlDirectory = required(state.controlDirectory, "controlDirectory");
  const memoryBytes = required(state.memoryBytes, "memoryBytes");
  const maxProcesses = required(state.maxProcesses, "maxProcesses");
  const tmpfsBytes = Math.max(1, Math.min(Math.floor(memoryBytes / 4), 536_870_912));
  const hostMounts = [
    {
      Type: "bind",
      Source: sourceDirectory,
      Target: CPP_CUTE_AOT_CONTAINER_SOURCE_ROOT,
      ReadOnly: true,
      BindOptions: { Propagation: "rprivate" },
    },
    {
      Type: "bind",
      Source: controlDirectory,
      Target: CPP_CUTE_AOT_CONTAINER_CONTROL_ROOT,
      ReadOnly: true,
      BindOptions: { Propagation: "rprivate" },
    },
  ];
  const realizedMounts = [
    {
      Type: "bind",
      Source: sourceDirectory,
      Destination: CPP_CUTE_AOT_CONTAINER_SOURCE_ROOT,
      Mode: "",
      RW: false,
      Propagation: "rprivate",
    },
    {
      Type: "bind",
      Source: controlDirectory,
      Destination: CPP_CUTE_AOT_CONTAINER_CONTROL_ROOT,
      Mode: "",
      RW: false,
      Propagation: "rprivate",
    },
  ];
  return {
    args: [...CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.arguments],
    config: {
      Hostname: CPP_CUTE_AOT_CONTAINER_HOSTNAME,
      Domainname: "",
      User: `${CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.user.uid}:${CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.user.gid}`,
      AttachStdin: false,
      AttachStdout: true,
      AttachStderr: true,
      ExposedPorts: {},
      Tty: false,
      OpenStdin: false,
      StdinOnce: false,
      Env: [],
      Cmd: [...CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.arguments],
      Healthcheck: { Test: ["NONE"] },
      Image: prepared.authorized.imageReference,
      Volumes: {},
      WorkingDir: CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.workingDirectory,
      Entrypoint: [CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.entrypoint],
      Labels: containerLabels(state),
    },
    hostConfig: {
      Binds: [],
      LogConfig: { Type: "none", Config: {} },
      NetworkMode: "none",
      PortBindings: {},
      RestartPolicy: { Name: "no", MaximumRetryCount: 0 },
      AutoRemove: false,
      VolumeDriver: "",
      VolumesFrom: [],
      ConsoleSize: [0, 0],
      CapAdd: [],
      CapDrop: ["ALL"],
      CgroupnsMode: "private",
      Dns: [],
      DnsOptions: [],
      DnsSearch: [],
      ExtraHosts: [],
      GroupAdd: [],
      IpcMode: "none",
      Cgroup: "",
      Links: [],
      OomScoreAdj: 0,
      PidMode: "",
      Privileged: false,
      PublishAllPorts: false,
      ReadonlyRootfs: true,
      SecurityOpt: ["no-new-privileges=true"],
      StorageOpt: {},
      Tmpfs: { "/tmp": `rw,noexec,nosuid,nodev,size=${tmpfsBytes},mode=1777` },
      UTSMode: "",
      UsernsMode: "",
      ShmSize: 67_108_864,
      Runtime: CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.runtime,
      Isolation: "",
      CpuShares: 0,
      Memory: memoryBytes,
      NanoCpus: 0,
      CgroupParent: "",
      ContainerIDFile: required(state.cidFile, "cidFile"),
      BlkioWeight: 0,
      BlkioWeightDevice: [],
      BlkioDeviceReadBps: [],
      BlkioDeviceWriteBps: [],
      BlkioDeviceReadIOps: [],
      BlkioDeviceWriteIOps: [],
      CpuPeriod: 0,
      CpuQuota: 0,
      CpuRealtimePeriod: 0,
      CpuRealtimeRuntime: 0,
      CpusetCpus: "",
      CpusetMems: "",
      Devices: [],
      DeviceCgroupRules: [],
      DeviceRequests: [],
      MemoryReservation: 0,
      MemorySwap: memoryBytes,
      MemorySwappiness: null,
      OomKillDisable: null,
      PidsLimit: maxProcesses,
      Ulimits: [],
      CpuCount: 0,
      CpuPercent: 0,
      IOMaximumIOps: 0,
      IOMaximumBandwidth: 0,
      MaskedPaths: [],
      ReadonlyPaths: [],
      Mounts: hostMounts,
    },
    id: containerId,
    image: prepared.fixture.manifestDigest,
    imageManifestDescriptor: {
      digest: prepared.fixture.manifestDigest,
      mediaType: prepared.metadata.manifest.mediaType,
      size: prepared.metadata.manifest.size,
      platform: { os: "linux", architecture: "amd64" },
    },
    mounts: realizedMounts,
    name: `/${required(state.containerName, "containerName")}`,
    path: CPP_CUTE_AOT_SANDBOX_POLICY_V1.process.entrypoint,
    restartCount: 0,
    schema: CPP_CUTE_AOT_DOCKER_CONTAINER_INSPECT_SCHEMA,
    state: {
      Status: status,
      Running: false,
      Paused: false,
      Restarting: false,
      OOMKilled: false,
      Dead: false,
      Pid: 0,
      ExitCode: 0,
      Error: "",
      StartedAt: status === "created" ? "0001-01-01T00:00:00Z" : "2026-07-16T00:00:00Z",
      FinishedAt: status === "created" ? "0001-01-01T00:00:00Z" : "2026-07-16T00:00:01Z",
    },
    version: 1,
  };
}

function snapshotStaging(
  prepared: PreparedLifecycleFixture,
  state: ContainerSessionState,
): StagingSnapshot {
  const inputs = copyCppCuteAotOfflineRunStagingInputs(prepared.plan);
  const runRoot = required(state.runRoot, "runRoot");
  const sourceDirectory = required(state.sourceDirectory, "sourceDirectory");
  const controlDirectory = required(state.controlDirectory, "controlDirectory");
  const paths = [
    runRoot,
    required(state.configDirectory, "configDirectory"),
    required(state.homeDirectory, "homeDirectory"),
    sourceDirectory,
    controlDirectory,
    join(controlDirectory, "profile.json"),
    join(controlDirectory, "job.json"),
    join(controlDirectory, "execution-environment.json"),
    ...inputs.sourceBlobs.map((source) => join(sourceDirectory, source.virtualPath.slice(1))),
  ];
  const bytes: Record<string, Uint8Array> = {
    profile: new Uint8Array(readFileSync(join(controlDirectory, "profile.json"))),
    job: new Uint8Array(readFileSync(join(controlDirectory, "job.json"))),
    environment: new Uint8Array(readFileSync(join(controlDirectory, "execution-environment.json"))),
  };
  for (const [index, source] of inputs.sourceBlobs.entries()) {
    bytes[`source-${index}`] = new Uint8Array(readFileSync(
      join(sourceDirectory, source.virtualPath.slice(1)),
    ));
  }
  return {
    modes: Object.fromEntries(paths.map((path) => [path, lstatSync(path).mode & 0o777])),
    bytes,
  };
}

function classifyRequest(
  request: BoundedChildProcessRequest,
  inspectCount: number,
): RequestKind {
  const operation = request.arguments.slice(4);
  if (operation[0] === "version") return "version";
  if (operation[0] === "info") return "info";
  if (operation[0] === "image" && operation[1] === "inspect") return "image";
  if (operation[0] !== "container") throw new Error(`unexpected request ${operation.join(" ")}`);
  if (operation[1] === "create") return "create";
  if (operation[1] === "start") return "start";
  if (operation[1] === "rm") return "remove";
  if (operation[1] === "ls") return "absence";
  if (operation[1] === "inspect") {
    const reference = operation.at(-1) ?? "";
    if (reference.startsWith("browsergrad-cpp-cute-aot-")) return "recovery";
    return inspectCount === 0 ? "inspect-created" : "inspect-terminal";
  }
  throw new Error(`unexpected request ${operation.join(" ")}`);
}

function createAdapter(
  prepared: PreparedLifecycleFixture,
  options: AdapterOptions = {},
) {
  const requests: RecordedRequest[] = [];
  const state: ContainerSessionState = {};
  let staging: StagingSnapshot | undefined;
  let inspectCount = 0;
  const containerId = options.containerId ?? CONTAINER_ID;
  const adapter = vi.fn(async (
    request: BoundedChildProcessRequest,
  ): Promise<BoundedChildProcessResult> => {
    const kind = classifyRequest(request, inspectCount);
    requests.push({ kind, request });
    state.runRoot ??= request.cwd;
    state.configDirectory ??= argumentAfter(request.arguments, "--config");
    state.homeDirectory ??= required(request.environment.HOME, "HOME");
    switch (kind) {
      case "version": return jsonResult(versionProjection());
      case "info": return jsonResult(infoProjection());
      case "image": return jsonResult(imageProjection(prepared));
      case "create": {
        captureCreateState(request, state);
        staging = snapshotStaging(prepared, state);
        const cidFileId = options.cidFileId === undefined ? containerId : options.cidFileId;
        if (cidFileId !== null) writeFileSync(required(state.cidFile, "cidFile"), cidFileId);
        return processResult(`${options.createStdoutId ?? containerId}\n`);
      }
      case "recovery": {
        if (options.recoveryBytes !== undefined) return processResult(options.recoveryBytes);
        const projection = recoveryProjection(state, containerId);
        options.mutateRecovery?.(projection);
        return jsonResult(projection);
      }
      case "inspect-created": {
        inspectCount += 1;
        const projection = containerProjection(prepared, state, containerId, "created");
        options.mutateCreated?.(projection);
        return jsonResult(projection);
      }
      case "start": return options.startResult ?? processResult(options.frameBytes ?? prepared.frame);
      case "inspect-terminal": {
        inspectCount += 1;
        const projection = containerProjection(prepared, state, containerId, "exited");
        options.mutateTerminal?.(projection);
        return jsonResult(projection);
      }
      case "remove": {
        state.removeForced = request.arguments.includes("--force");
        return options.removeResult ?? processResult(`${containerId}\n`);
      }
      case "absence": return options.absenceResult ?? processResult();
    }
  });
  return {
    adapter,
    requests,
    state,
    staging: () => staging,
  };
}

function expectRunRootRemoved(state: ContainerSessionState): void {
  expect(existsSync(required(state.runRoot, "runRoot"))).toBe(false);
}

function cleanPreservedRunRoot(state: ContainerSessionState): void {
  const runRoot = required(state.runRoot, "runRoot");
  expect(existsSync(runRoot)).toBe(true);
  makeTreeOwnerWritable(runRoot);
  rmSync(runRoot, { recursive: true, force: true });
}

function makeTreeOwnerWritable(path: string): void {
  const status = lstatSync(path);
  if (!status.isDirectory() || status.isSymbolicLink()) return;
  chmodSync(path, (status.mode & 0o777) | 0o700);
  for (const child of readdirSync(path)) makeTreeOwnerWritable(join(path, child));
}

function expectClosedRequests(records: readonly RecordedRequest[]): void {
  const runRoot = records[0]?.request.cwd;
  expect(runRoot).toMatch(/^\/tmp\/browsergrad-cpp-cute-docker-/u);
  for (const { request } of records) {
    expect(request.cwd).toBe(runRoot);
    expect(Object.fromEntries(Object.entries(request.environment))).toEqual({
      DOCKER_API_VERSION: "1.49",
      HOME: join(required(runRoot, "runRoot"), "home"),
      LANG: "C",
      LC_ALL: "C",
      TZ: "UTC",
    });
    expect(request.arguments.slice(0, 4)).toEqual([
      "--config",
      join(required(runRoot, "runRoot"), "docker-config"),
      "--host=unix:///var/run/docker.sock",
      "--log-level=error",
    ]);
  }
}

describe("C++/CuTe AOT Docker lifecycle", () => {
  it("accepts one exact execution, stages immutable canonical bytes, cleans, and separates authority", async () => {
    const prepared = await prepareFixture();
    const harness = createAdapter(prepared);
    const completed = await __executeCppCuteAotDockerRunWithProcessForTest(
      prepared.authorized,
      harness.adapter,
    );

    expect(completed).toMatchObject({
      jobId: prepared.authorized.jobId,
      profileHash: prepared.authorized.profileHash,
      executionPlanSha256: prepared.authorized.executionPlanSha256,
      containerId: CONTAINER_ID,
      imageId: prepared.fixture.manifestDigest,
      frontendOutcome: "accepted",
    });
    expect(harness.requests.map(({ kind }) => kind)).toEqual([
      "version", "info", "image", "create", "inspect-created", "start",
      "inspect-terminal", "remove", "absence",
    ]);
    expectClosedRequests(harness.requests);
    expect(harness.state.removeForced).toBe(false);
    expectRunRootRemoved(harness.state);

    const staging = required(harness.staging(), "staging snapshot");
    const inputs = copyCppCuteAotOfflineRunStagingInputs(prepared.plan);
    expect(staging.modes[required(harness.state.runRoot, "runRoot")]).toBe(0o700);
    expect(staging.modes[required(harness.state.configDirectory, "configDirectory")]).toBe(0o700);
    expect(staging.modes[required(harness.state.homeDirectory, "homeDirectory")]).toBe(0o700);
    expect(staging.modes[required(harness.state.sourceDirectory, "sourceDirectory")]).toBe(0o555);
    expect(staging.modes[required(harness.state.controlDirectory, "controlDirectory")]).toBe(0o555);
    expect(staging.modes[join(required(harness.state.controlDirectory, "controlDirectory"), "profile.json")]).toBe(0o444);
    expect(staging.modes[join(required(harness.state.controlDirectory, "controlDirectory"), "job.json")]).toBe(0o444);
    expect(staging.modes[join(required(harness.state.controlDirectory, "controlDirectory"), "execution-environment.json")]).toBe(0o444);
    expect(staging.bytes.profile).toEqual(inputs.profileBytes);
    expect(staging.bytes.job).toEqual(inputs.jobBytes);
    expect(staging.bytes.environment).toEqual(inputs.environmentBytes);
    for (const [index, source] of inputs.sourceBlobs.entries()) {
      const sourcePath = join(required(harness.state.sourceDirectory, "sourceDirectory"), source.virtualPath.slice(1));
      expect(staging.modes[sourcePath]).toBe(0o444);
      expect(staging.bytes[`source-${index}`]).toEqual(source.bytes);
    }

    const common = {
      runRoot: required(harness.state.runRoot, "runRoot"),
      configDirectory: required(harness.state.configDirectory, "configDirectory"),
      homeDirectory: required(harness.state.homeDirectory, "homeDirectory"),
    };
    const plan = prepared.plan;
    const limits = unwrapDistProfile(unwrapDistOfflineRun(plan).profile).profile.extractionLimits;
    const expectedRequests = [
      buildCppCuteAotDockerVersionRequest(common),
      buildCppCuteAotDockerInfoRequest(common),
      buildCppCuteAotDockerImageInspectRequest({ ...common, imageReference: plan.imageReference }),
      buildCppCuteAotDockerCreateRequest({
        ...common,
        controlDirectory: required(harness.state.controlDirectory, "controlDirectory"),
        sourceDirectory: required(harness.state.sourceDirectory, "sourceDirectory"),
        containerIdFile: required(harness.state.cidFile, "cidFile"),
        containerName: required(harness.state.containerName, "containerName"),
        sessionNonce: required(harness.state.sessionNonce, "sessionNonce"),
        imageReference: plan.imageReference,
        jobId: plan.jobId,
        executionPlanSha256: plan.executionPlanSha256,
        memoryBytes: limits.maxMemoryBytes,
        maxProcesses: limits.maxProcesses,
      }),
      buildCppCuteAotDockerContainerInspectRequest({ ...common, containerId: CONTAINER_ID }),
      buildCppCuteAotDockerStartAttachedRequest({
        ...common,
        containerId: CONTAINER_ID,
        timeoutMs: harness.requests[5]?.request.timeoutMs ?? 1,
        stdoutByteLimit: plan.frameByteLimit,
      }),
      buildCppCuteAotDockerContainerInspectRequest({ ...common, containerId: CONTAINER_ID }),
      buildCppCuteAotDockerRemoveRequest({ ...common, containerId: CONTAINER_ID, force: false }),
      buildCppCuteAotDockerAbsenceRequest({ ...common, containerId: CONTAINER_ID }),
    ];
    for (const [index, expectedRequest] of expectedRequests.entries()) {
      const actual = harness.requests[index]?.request;
      expect(actual?.arguments).toEqual(expectedRequest.arguments);
      expect(actual?.executable).toBe(expectedRequest.executable);
      expect(actual?.stdoutByteLimit).toBe(expectedRequest.stdoutByteLimit);
      expect(actual?.stderrByteLimit).toBe(expectedRequest.stderrByteLimit);
      expect(actual?.killGraceMs).toBe(expectedRequest.killGraceMs);
      expect(actual?.timeoutMs).toBeGreaterThan(0);
      expect(actual?.timeoutMs).toBeLessThanOrEqual(expectedRequest.timeoutMs);
    }

    expect(__unwrapCompletedCppCuteAotDockerRunForTest(completed).result.frontendOutcome).toBe("accepted");
    expect(() => unwrapCompletedCppCuteAotDockerRun(completed)).toThrow(/DOCKER-RUN-UNVERIFIED/u);
    expect(() => __unwrapCompletedCppCuteAotDockerRunForTest(structuredClone(completed)))
      .toThrow(/DOCKER-RUN-UNVERIFIED/u);
  });

  it("recovers exact identity after create ID mismatch, then force-cleans without starting", async () => {
    const prepared = await prepareFixture();
    const harness = createAdapter(prepared, { createStdoutId: OTHER_CONTAINER_ID });

    await expect(__executeCppCuteAotDockerRunWithProcessForTest(prepared.authorized, harness.adapter))
      .rejects.toThrow(/DOCKER-RUN-CREATE/u);
    expect(harness.requests.map(({ kind }) => kind)).toEqual([
      "version", "info", "image", "create", "recovery", "inspect-created", "remove", "absence",
    ]);
    expect(harness.state.removeForced).toBe(true);
    expectRunRootRemoved(harness.state);
  });

  it.each([
    {
      name: "forged labels",
      options: {
        createStdoutId: OTHER_CONTAINER_ID,
        mutateRecovery: (projection: Record<string, unknown>) => {
          (projection.labels as Record<string, unknown>)["browsergrad.session"] = "0".repeat(32);
        },
      } satisfies AdapterOptions,
    },
    {
      name: "malformed JSON",
      options: {
        createStdoutId: OTHER_CONTAINER_ID,
        recoveryBytes: encoder.encode("{\n"),
      } satisfies AdapterOptions,
    },
  ])("fail-stops and preserves staging for $name recovery evidence", async ({ options }) => {
    const prepared = await prepareFixture();
    const harness = createAdapter(prepared, options);
    try {
      await expect(__executeCppCuteAotDockerRunWithProcessForTest(prepared.authorized, harness.adapter))
        .rejects.toThrow(/DOCKER-RUN-CLEANUP/u);
      expect(harness.requests.map(({ kind }) => kind)).toEqual([
        "version", "info", "image", "create", "recovery",
      ]);
      expect(existsSync(required(harness.state.runRoot, "runRoot"))).toBe(true);
    } finally {
      cleanPreservedRunRoot(harness.state);
    }
  });

  it("force-cleans a tampered created projection and never starts it", async () => {
    const prepared = await prepareFixture();
    const harness = createAdapter(prepared, {
      mutateCreated: (projection) => {
        (projection.config as Record<string, unknown>).User = "0:0";
      },
    });

    await expect(__executeCppCuteAotDockerRunWithProcessForTest(prepared.authorized, harness.adapter))
      .rejects.toThrow(/DOCKER-RUN-CONTAINER-MISMATCH/u);
    expect(harness.requests.map(({ kind }) => kind)).toEqual([
      "version", "info", "image", "create", "inspect-created", "remove", "absence",
    ]);
    expect(harness.state.removeForced).toBe(true);
    expectRunRootRemoved(harness.state);
  });

  it("force-cleans after attached start failure", async () => {
    const prepared = await prepareFixture();
    const harness = createAdapter(prepared, {
      startResult: processResult(new Uint8Array(), { exitCode: 17 }),
    });

    await expect(__executeCppCuteAotDockerRunWithProcessForTest(prepared.authorized, harness.adapter))
      .rejects.toThrow(/DOCKER-RUN-START/u);
    expect(harness.requests.map(({ kind }) => kind)).toEqual([
      "version", "info", "image", "create", "inspect-created", "start",
      "inspect-terminal", "remove", "absence",
    ]);
    expect(harness.state.removeForced).toBe(true);
    expectRunRootRemoved(harness.state);
  });

  it("accepts bounded successful supervisor diagnostics on stderr", async () => {
    const prepared = await prepareFixture();
    const harness = createAdapter(prepared, {
      startResult: processResult(prepared.frame, { stderr: encoder.encode("bounded diagnostic\n") }),
    });

    await expect(__executeCppCuteAotDockerRunWithProcessForTest(
      prepared.authorized,
      harness.adapter,
    )).resolves.toMatchObject({ frontendOutcome: "accepted" });
    expect(harness.state.removeForced).toBe(false);
    expectRunRootRemoved(harness.state);
  });

  it("force-cleans a terminal-state mismatch and retains the failure", async () => {
    const prepared = await prepareFixture();
    const harness = createAdapter(prepared, {
      mutateTerminal: (projection) => {
        (projection.state as Record<string, unknown>).FinishedAt = "2026-07-15T23:59:59Z";
      },
    });

    await expect(__executeCppCuteAotDockerRunWithProcessForTest(prepared.authorized, harness.adapter))
      .rejects.toThrow(/DOCKER-RUN-CONTAINER-MISMATCH/u);
    expect(harness.requests.slice(-2).map(({ kind }) => kind)).toEqual(["remove", "absence"]);
    expect(harness.state.removeForced).toBe(true);
    expectRunRootRemoved(harness.state);
  });

  it("retains primary and cleanup failures after proving absence", async () => {
    const prepared = await prepareFixture();
    const harness = createAdapter(prepared, {
      startResult: processResult(new Uint8Array(), { exitCode: 17 }),
      removeResult: processResult(new Uint8Array(), { exitCode: 1 }),
    });

    let failure: unknown;
    try {
      await __executeCppCuteAotDockerRunWithProcessForTest(prepared.authorized, harness.adapter);
    } catch (error) {
      failure = error;
    }
    expect(failure).toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-RUN-CLEANUP" });
    expect((failure as Error).cause).toBeInstanceOf(AggregateError);
    expect(((failure as Error).cause as AggregateError).errors).toHaveLength(2);
    expectRunRootRemoved(harness.state);
  });

  it("rejects a malformed result frame only after container and staging cleanup", async () => {
    const prepared = await prepareFixture();
    const harness = createAdapter(prepared, { frameBytes: encoder.encode("not-a-result-frame") });

    await expect(__executeCppCuteAotDockerRunWithProcessForTest(prepared.authorized, harness.adapter))
      .rejects.toThrow(/AOT-RUNNER-FRAME-INVALID/u);
    expect(harness.requests.slice(-2).map(({ kind }) => kind)).toEqual(["remove", "absence"]);
    expect(harness.state.removeForced).toBe(false);
    expectRunRootRemoved(harness.state);
  });

  it("reports removal failure after proving absence and removes staging", async () => {
    const prepared = await prepareFixture();
    const harness = createAdapter(prepared, {
      removeResult: processResult(new Uint8Array(), { exitCode: 1 }),
    });

    await expect(__executeCppCuteAotDockerRunWithProcessForTest(prepared.authorized, harness.adapter))
      .rejects.toThrow(/DOCKER-RUN-CLEANUP/u);
    expect(harness.requests.slice(-2).map(({ kind }) => kind)).toEqual(["remove", "absence"]);
    expect(harness.state.removeForced).toBe(false);
    expectRunRootRemoved(harness.state);
  });

  it("preserves private staging when absence cannot be proved", async () => {
    const prepared = await prepareFixture();
    const harness = createAdapter(prepared, {
      absenceResult: processResult(`${CONTAINER_ID}\n`),
    });
    try {
      await expect(__executeCppCuteAotDockerRunWithProcessForTest(prepared.authorized, harness.adapter))
        .rejects.toThrow(/DOCKER-RUN-CLEANUP/u);
      expect(harness.requests.slice(-2).map(({ kind }) => kind)).toEqual(["remove", "absence"]);
      expect(existsSync(required(harness.state.runRoot, "runRoot"))).toBe(true);
    } finally {
      cleanPreservedRunRoot(harness.state);
    }
  });

  it("rejects attacker-controlled lifecycle builder identities and fields", () => {
    const common = {
      runRoot: "/tmp/browsergrad-cpp-cute-docker-abcdefghijkl",
      configDirectory: "/tmp/browsergrad-cpp-cute-docker-abcdefghijkl/docker-config",
      homeDirectory: "/tmp/browsergrad-cpp-cute-docker-abcdefghijkl/home",
    };
    expect(() => buildCppCuteAotDockerContainerInspectRequest({
      ...common,
      containerId: "short",
    })).toThrow(/DOCKER-PROCESS-INVALID/u);
    expect(() => buildCppCuteAotDockerStartAttachedRequest({
      ...common,
      containerId: CONTAINER_ID,
      timeoutMs: 1,
      stdoutByteLimit: 1,
      attacker: true,
    } as never)).toThrow(/DOCKER-PROCESS-INVALID/u);
    expect(() => buildCppCuteAotDockerRemoveRequest({
      ...common,
      containerId: "A".repeat(64),
      force: true,
    })).toThrow(/DOCKER-PROCESS-INVALID/u);

    const minimumMemoryCreate = buildCppCuteAotDockerCreateRequest({
      ...common,
      controlDirectory: `${common.runRoot}/control`,
      sourceDirectory: `${common.runRoot}/source`,
      containerIdFile: `${common.runRoot}/container.cid`,
      containerName: `browsergrad-cpp-cute-aot-${"0".repeat(32)}`,
      sessionNonce: "0".repeat(32),
      imageReference: `registry.example.com/browsergrad/cpp-cute@sha256:${"a".repeat(64)}`,
      jobId: `bg.cpp.aot-job.sha256.${"b".repeat(64)}`,
      executionPlanSha256: "c".repeat(64),
      memoryBytes: 1,
      maxProcesses: 1,
    });
    expect(minimumMemoryCreate.arguments).toContain(
      "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=1,mode=1777",
    );
  });
});
