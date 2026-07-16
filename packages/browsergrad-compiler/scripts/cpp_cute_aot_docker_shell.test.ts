import {
  existsSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import process from "node:process";

import { beforeEach, describe, expect, it, vi } from "vitest";

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
import { prepareCppCuteAotOfflineRun as prepareDistOfflineRun } from "../dist/cpp_cute_aot_runner_plan.js";
import { prepareCppCuteFrontendProfile as prepareDistProfile } from "../dist/cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_AOT_DOCKER_API_VERSION,
  CPP_CUTE_AOT_DOCKER_EXECUTABLE,
  CPP_CUTE_AOT_DOCKER_INFO_FORMAT,
  CPP_CUTE_AOT_DOCKER_INFO_LIMITS,
  CPP_CUTE_AOT_DOCKER_INFO_SCHEMA,
  CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_FORMAT,
  CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_LIMITS,
  CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_SCHEMA,
  CPP_CUTE_AOT_DOCKER_VERSION_FORMAT,
  CPP_CUTE_AOT_DOCKER_VERSION_LIMITS,
  CPP_CUTE_AOT_DOCKER_VERSION_SCHEMA,
} from "../dist/cpp_cute_aot_policy.js";
import {
  buildCppCuteAotDockerInfoRequest,
  buildCppCuteAotDockerImageInspectRequest,
  buildCppCuteAotDockerVersionRequest,
  runBoundedChildProcess,
  type BoundedChildProcessRequest,
  type BoundedChildProcessResult,
} from "./cpp_cute_aot_docker_process.mjs";
import {
  __observeCppCuteAotLocalDockerImageWithProcessForTest,
  __unwrapObservedCppCuteAotLocalDockerImageForTest,
  unwrapObservedCppCuteAotLocalDockerImage,
} from "./cpp_cute_aot_docker_shell.mjs";
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
  DEFAULT_DIFF_ID,
  defaultLayer,
  type CppCuteAotOciFixture,
} from "../tests/compiler/support/cpp_cute_aot_oci_fixtures.js";

const encoder = new TextEncoder();

interface PreparedDockerFixture {
  readonly fixture: CppCuteAotOciFixture;
  readonly metadata: VerifiedCppCuteAotOciMetadata;
  readonly authorized: AuthorizedCppCuteAotOciMetadata;
}

interface MutableProjection {
  config: unknown;
  descriptor: { digest: string; mediaType: string; size: number };
  id: string;
  platform: { architecture: string; os: string; osVersion: string; variant: string };
  repoDigests: string[];
  rootfs: { diffIds: string[]; type: string };
  schema: string;
  version: number;
  [key: string]: unknown;
}

interface MutableVersionProjection {
  client: {
    apiVersion: string;
    defaultApiVersion: string;
    version: string;
  };
  schema: string;
  server: {
    apiVersion: string;
    arch: string;
    minApiVersion: string;
    os: string;
    version: string;
  };
  version: number;
  [key: string]: unknown;
}

interface MutableInfoProjection {
  architecture: string;
  driverStatus: string[][];
  osType: string;
  schema: string;
  serverVersion: string;
  version: number;
  [key: string]: unknown;
}

type DockerProbe = "version" | "info" | "image";
const DOCKER_PROBE_SEQUENCE = ["version", "info", "image"] as const;

interface ProbeAdapterOptions {
  readonly mutateVersion?: (value: MutableVersionProjection) => void;
  readonly mutateInfo?: (value: MutableInfoProjection) => void;
  readonly mutateImage?: (value: MutableProjection) => void;
  readonly resultFor?: (
    probe: DockerProbe,
    value: MutableVersionProjection | MutableInfoProjection | MutableProjection,
    request: BoundedChildProcessRequest,
  ) => BoundedChildProcessResult;
  readonly afterProbe?: (
    probe: DockerProbe,
    request: BoundedChildProcessRequest,
  ) => void;
}

function processRequest(overrides: Partial<BoundedChildProcessRequest> = {}): BoundedChildProcessRequest {
  return {
    executable: process.execPath,
    arguments: ["-e", "process.stdout.write('ok')"],
    cwd: process.cwd(),
    environment: { LANG: "C" },
    timeoutMs: 2_000,
    killGraceMs: 1_000,
    stdoutByteLimit: 1_024,
    stderrByteLimit: 1_024,
    ...overrides,
  };
}

function versionProjection(): MutableVersionProjection {
  return {
    client: {
      apiVersion: "1.49",
      defaultApiVersion: "1.55",
      version: "29.6.1",
    },
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

function infoProjection(): MutableInfoProjection {
  return {
    architecture: "x86_64",
    driverStatus: [["driver-type", "io.containerd.snapshotter.v1"]],
    osType: "linux",
    schema: CPP_CUTE_AOT_DOCKER_INFO_SCHEMA,
    serverVersion: "29.6.1",
    version: 1,
  };
}

async function prepareFixture(twoLayers = false): Promise<PreparedDockerFixture> {
  const secondDiffId = `sha256:${"c".repeat(64)}`;
  const fixture = await createCppCuteAotOciFixture(twoLayers
    ? {
        layers: [defaultLayer(), defaultLayer({ digest: `sha256:${"d".repeat(64)}` })],
        diffIds: [DEFAULT_DIFF_ID, secondDiffId],
        environmentMatchesOciLayers: true,
      }
    : {});
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
  };
}

function projection(prepared: PreparedDockerFixture): MutableProjection {
  const rawConfig = JSON.parse(new TextDecoder().decode(prepared.fixture.evidence.configBytes)) as {
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
    rootfs: { diffIds: rawConfig.rootfs.diff_ids, type: "layers" },
    schema: CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_SCHEMA,
    version: 1,
  };
}

function resultFromProjection(value: unknown): BoundedChildProcessResult {
  return Object.freeze({
    exitCode: 0,
    signal: null,
    stdout: encoder.encode(`${JSON.stringify(value)}\n`),
    stderr: new Uint8Array(),
  });
}

function probeFromRequest(request: BoundedChildProcessRequest): DockerProbe {
  if (request.arguments.includes("version")) return "version";
  if (request.arguments.includes("info")) return "info";
  const image = request.arguments.indexOf("image");
  if (image >= 0 && request.arguments[image + 1] === "inspect") return "image";
  throw new Error(`unexpected Docker request: ${request.arguments.join(" ")}`);
}

function createProbeAdapter(
  prepared: PreparedDockerFixture,
  options: ProbeAdapterOptions = {},
) {
  const requests: BoundedChildProcessRequest[] = [];
  const adapter = vi.fn(async (request: BoundedChildProcessRequest) => {
    const probe = probeFromRequest(request);
    expect(probe).toBe(DOCKER_PROBE_SEQUENCE[requests.length]);
    requests.push(request);
    let value: MutableVersionProjection | MutableInfoProjection | MutableProjection;
    if (probe === "version") {
      value = versionProjection();
      options.mutateVersion?.(value);
    } else if (probe === "info") {
      value = infoProjection();
      options.mutateInfo?.(value);
    } else {
      value = projection(prepared);
      options.mutateImage?.(value);
    }
    const result = options.resultFor?.(probe, value, request) ?? resultFromProjection(value);
    options.afterProbe?.(probe, request);
    return result;
  });
  return { adapter, requests };
}

async function waitForFile(path: string): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (existsSync(path)) return;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error(`timed out waiting for child PID file ${path}`);
}

function expectProcessGone(pid: number): void {
  try {
    process.kill(pid, 0);
  } catch (error) {
    expect((error as NodeJS.ErrnoException).code).toBe("ESRCH");
    return;
  }
  throw new Error(`bounded child process ${pid} is still alive`);
}

describe("C++/CuTe AOT bounded Docker process", () => {
  it("captures bounded bytes with a closed environment and no shell", async () => {
    const previous = process.env.BG_AMBIENT_SECRET;
    process.env.BG_AMBIENT_SECRET = "must-not-be-inherited";
    try {
      const result = await runBoundedChildProcess(processRequest({
        arguments: [
          "-e",
          "process.stdout.write(`${process.env.LANG}|${process.env.BG_AMBIENT_SECRET ?? ''}|${process.argv[1]}`)",
          "; echo injected",
        ],
        environment: { LANG: "C" },
      }));
      expect(new TextDecoder().decode(result.stdout)).toBe("C||; echo injected");
      expect(result.stderr.byteLength).toBe(0);
      expect(result.exitCode).toBe(0);
      expect(result.signal).toBeNull();
    } finally {
      if (previous === undefined) delete process.env.BG_AMBIENT_SECRET;
      else process.env.BG_AMBIENT_SECRET = previous;
    }
  });

  it.each([
    ["stdout", "process.stdout", "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-STDOUT-LIMIT"],
    ["stderr", "process.stderr", "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-STDERR-LIMIT"],
  ])("accepts the exact %s cap and rejects one byte more", async (stream, target, code) => {
    const exact = await runBoundedChildProcess(processRequest({
      arguments: ["-e", `${target}.write(Buffer.alloc(16))`],
      stdoutByteLimit: 16,
      stderrByteLimit: 16,
    }));
    expect(stream === "stdout" ? exact.stdout.byteLength : exact.stderr.byteLength).toBe(16);

    await expect(runBoundedChildProcess(processRequest({
      arguments: ["-e", `for(let i=0;i<17;i+=1)${target}.write(Buffer.alloc(1))`],
      stdoutByteLimit: 16,
      stderrByteLimit: 16,
    }))).rejects.toMatchObject({ code });
  });

  it("drains simultaneous chunked stdout and stderr without deadlock", async () => {
    const result = await runBoundedChildProcess(processRequest({
      arguments: [
        "-e",
        "for(let i=0;i<128;i+=1){process.stdout.write(Buffer.alloc(8,65));process.stderr.write(Buffer.alloc(8,66))}",
      ],
      stdoutByteLimit: 1_024,
      stderrByteLimit: 1_024,
    }));
    expect(result.stdout.byteLength).toBe(1_024);
    expect(result.stderr.byteLength).toBe(1_024);
  });

  it.each(["timeout", "cancellation"] as const)("kills and reaps %s", async (reason) => {
    const root = mkdtempSync(join(tmpdir(), "browsergrad-bounded-child-"));
    const pidFile = join(root, "pid");
    const controller = new AbortController();
    const pending = runBoundedChildProcess(processRequest({
      arguments: [
        "-e",
        "require('node:fs').writeFileSync(process.argv[1],String(process.pid));setInterval(()=>{},1000)",
        pidFile,
      ],
      timeoutMs: reason === "timeout" ? 500 : 2_000,
      signal: controller.signal,
    }));

    try {
      await waitForFile(pidFile);
      if (reason === "cancellation") controller.abort(new Error("must not leak"));
      await expect(pending).rejects.toMatchObject({
        code: reason === "timeout"
          ? "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-TIMEOUT"
          : "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-CANCELLED",
        message: expect.not.stringContaining("must not leak"),
      });
      expectProcessGone(Number(readFileSync(pidFile, "utf8")));
    } finally {
      controller.abort();
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("classifies spawn failure and preserves clean nonzero exit", async () => {
    await expect(runBoundedChildProcess(processRequest({
      executable: "/definitely/not/a/real/browsergrad-executable",
    }))).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-SPAWN" });
    await expect(runBoundedChildProcess(processRequest({
      arguments: ["-e", "process.exit(7)"],
    }))).resolves.toMatchObject({ exitCode: 7, signal: null });
  });

  it("builds the exact policy-owned version, info, then image-inspect requests", () => {
    const root = "/tmp/browsergrad-cpp-cute-docker-fixture";
    const imageReference = `ghcr.io/unlocalhosted/browsergrad-cpp-cute-aot@sha256:${"a".repeat(64)}`;
    const common = {
      runRoot: root,
      configDirectory: `${root}/docker-config`,
      homeDirectory: `${root}/home`,
    } as const;
    const environment = {
      DOCKER_API_VERSION: CPP_CUTE_AOT_DOCKER_API_VERSION,
      HOME: `${root}/home`,
      LANG: "C",
      LC_ALL: "C",
      TZ: "UTC",
    };
    const prefix = [
      "--config", `${root}/docker-config`,
      "--host=unix:///var/run/docker.sock",
      "--log-level=error",
    ];
    const cases = [
      {
        request: buildCppCuteAotDockerVersionRequest(common),
        arguments: [...prefix, "version", "--format", CPP_CUTE_AOT_DOCKER_VERSION_FORMAT],
        limits: CPP_CUTE_AOT_DOCKER_VERSION_LIMITS,
      },
      {
        request: buildCppCuteAotDockerInfoRequest(common),
        arguments: [...prefix, "info", "--format", CPP_CUTE_AOT_DOCKER_INFO_FORMAT],
        limits: CPP_CUTE_AOT_DOCKER_INFO_LIMITS,
      },
      {
        request: buildCppCuteAotDockerImageInspectRequest({ ...common, imageReference }),
        arguments: [
          ...prefix,
          "image", "inspect",
          "--platform=linux/amd64",
          "--format", CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_FORMAT,
          imageReference,
        ],
        limits: CPP_CUTE_AOT_DOCKER_IMAGE_INSPECT_LIMITS,
      },
    ];
    for (const { request, arguments: requestArguments, limits } of cases) {
      expect(request).toEqual({
        executable: CPP_CUTE_AOT_DOCKER_EXECUTABLE,
        arguments: requestArguments,
        cwd: root,
        environment,
        timeoutMs: limits.timeoutMs,
        killGraceMs: limits.killGraceMs,
        stdoutByteLimit: limits.stdoutBytes,
        stderrByteLimit: limits.stderrBytes,
      });
      expect(Object.isFrozen(request)).toBe(true);
      expect(Object.isFrozen(request.arguments)).toBe(true);
      expect(Object.isFrozen(request.environment)).toBe(true);
      expect(Object.keys(request.environment).sort()).toEqual([
        "DOCKER_API_VERSION", "HOME", "LANG", "LC_ALL", "TZ",
      ]);
    }
    expect(cases[2]?.request.arguments.at(-1)).toBe(imageReference);
    expect(cases[2]?.request.arguments).not.toContain(`sha256:${"b".repeat(64)}`);
  });

  it("rejects caller-controlled paths, image tags, args, and malformed requests", () => {
    const root = "/tmp/browsergrad-cpp-cute-docker-fixture";
    expect(() => buildCppCuteAotDockerImageInspectRequest({
      runRoot: "/var/tmp/escape",
      configDirectory: "/var/tmp/escape/docker-config",
      homeDirectory: "/var/tmp/escape/home",
      imageReference: `ghcr.io/u/i@sha256:${"a".repeat(64)}`,
    })).toThrow(/PROCESS-INVALID/u);
    expect(() => buildCppCuteAotDockerImageInspectRequest({
      runRoot: root,
      configDirectory: `${root}/other`,
      homeDirectory: `${root}/home`,
      imageReference: "ghcr.io/u/i:latest",
    })).toThrow(/PROCESS-INVALID/u);
    expect(() => runBoundedChildProcess({
      ...processRequest(),
      shell: true,
    } as BoundedChildProcessRequest)).toThrow(/PROCESS-INVALID/u);
  });

  it("rejects hostile request structures and pre-abort without spawning", async () => {
    const ownKeysProxy = new Proxy(processRequest(), {
      ownKeys: () => { throw new Error("attacker ownKeys"); },
    });
    expect(() => runBoundedChildProcess(ownKeysProxy)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-INVALID",
        message: expect.not.stringContaining("attacker ownKeys"),
      }),
    );

    const accessorArguments = ["-e", "process.exit(0)"];
    Object.defineProperty(accessorArguments, "0", {
      enumerable: true,
      get: () => { throw new Error("attacker getter"); },
    });
    expect(() => runBoundedChildProcess(processRequest({
      arguments: accessorArguments,
    }))).toThrowError(/PROCESS-INVALID/u);

    const accessorEnvironment: Record<string, string> = {};
    Object.defineProperty(accessorEnvironment, "LANG", {
      enumerable: true,
      get: () => { throw new Error("attacker environment"); },
    });
    expect(() => runBoundedChildProcess(processRequest({
      environment: accessorEnvironment,
    }))).toThrowError(/PROCESS-INVALID/u);

    const controller = new AbortController();
    controller.abort(new Error("must not leak"));
    await expect(runBoundedChildProcess(processRequest({
      executable: "/definitely/not/a/real/browsergrad-executable",
      signal: controller.signal,
    }))).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-PROCESS-CANCELLED",
      message: expect.not.stringContaining("must not leak"),
    });
  });
});

describe("C++/CuTe AOT local Docker image observation", () => {
  let prepared: PreparedDockerFixture;

  beforeEach(async () => {
    prepared = await prepareFixture(true);
  });

  it("cross-binds exact manifest, config lineage, platform, rootfs, and empty config", async () => {
    const { adapter, requests } = createProbeAdapter(prepared);
    const observed = await __observeCppCuteAotLocalDockerImageWithProcessForTest(
      prepared.authorized,
      adapter,
    );
    expect(observed).toEqual({
      jobId: prepared.authorized.jobId,
      profileHash: prepared.authorized.profileHash,
      executionPlanSha256: prepared.authorized.executionPlanSha256,
      imageReference: prepared.authorized.imageReference,
      manifestDigest: prepared.fixture.manifestDigest,
      imageId: prepared.fixture.manifestDigest,
      configDigest: prepared.fixture.configDigest,
      platform: "linux/amd64",
      dockerClientVersion: "29.6.1",
      dockerEngineVersion: "29.6.1",
      dockerRequestApiVersion: "1.49",
      dockerEngineApiVersion: "1.55",
      dockerEngineMinApiVersion: "1.40",
      dockerImageStore: "containerd",
      layerCount: 2,
      totalLayerBytes: 246,
    });
    expect(Object.isFrozen(observed)).toBe(true);
    expect(requests.map(probeFromRequest)).toEqual(["version", "info", "image"]);
    expect(requests[2]?.arguments.at(-1)).toBe(prepared.authorized.imageReference);
    expect(requests[2]?.arguments).not.toContain(prepared.authorized.configDigest);
    expect(requests.every((request) => request.cwd === requests[0]?.cwd)).toBe(true);
    expect(requests[0] === undefined ? true : existsSync(requests[0].cwd)).toBe(false);
    const record = __unwrapObservedCppCuteAotLocalDockerImageForTest(observed);
    expect(record.authorizedMetadata).toBe(prepared.authorized);
    expect(Object.isFrozen(record.repoDigests)).toBe(true);
    expect(() => unwrapObservedCppCuteAotLocalDockerImage(observed)).toThrow(/UNVERIFIED/u);
    expect(() => __unwrapObservedCppCuteAotLocalDockerImageForTest({ ...observed })).toThrow(/UNVERIFIED/u);
  });

  it("retains an additional canonical repository digest after exact-reference comparison", async () => {
    const additional = `other.example/team/image@${prepared.fixture.manifestDigest}`;
    const { adapter } = createProbeAdapter(prepared, {
      mutateImage: (value) => { value.repoDigests.push(additional); },
    });
    const observed = await __observeCppCuteAotLocalDockerImageWithProcessForTest(
      prepared.authorized,
      adapter,
    );
    expect(__unwrapObservedCppCuteAotLocalDockerImageForTest(observed).repoDigests).toEqual(
      [prepared.authorized.imageReference, additional].sort(),
    );
  });

  it.each([
    ["client version", (value: MutableVersionProjection) => { value.client.version = "29.6.0"; }],
    ["client request API", (value: MutableVersionProjection) => { value.client.apiVersion = "1.48"; }],
    ["client default API", (value: MutableVersionProjection) => { value.client.defaultApiVersion = "1.54"; }],
    ["engine version", (value: MutableVersionProjection) => { value.server.version = "29.6.0"; }],
    ["engine API", (value: MutableVersionProjection) => { value.server.apiVersion = "1.54"; }],
    ["engine minimum API", (value: MutableVersionProjection) => { value.server.minApiVersion = "1.39"; }],
    ["engine OS", (value: MutableVersionProjection) => { value.server.os = "windows"; }],
    ["engine architecture", (value: MutableVersionProjection) => { value.server.arch = "arm64"; }],
  ])("fails before info and image inspection on mismatched runtime %s", async (_name, mutate) => {
    const { adapter, requests } = createProbeAdapter(prepared, { mutateVersion: mutate });
    await expect(__observeCppCuteAotLocalDockerImageWithProcessForTest(
      prepared.authorized,
      adapter,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-MISMATCH" });
    expect(requests.map(probeFromRequest)).toEqual(["version"]);
    expect(requests[0] === undefined ? true : existsSync(requests[0].cwd)).toBe(false);
  });

  it.each([
    ["empty driver status", (value: MutableInfoProjection) => { value.driverStatus = []; }],
    ["legacy graphdriver store", (value: MutableInfoProjection) => { value.driverStatus = [["Backing Filesystem", "extfs"]]; }],
    ["additional driver status", (value: MutableInfoProjection) => {
      value.driverStatus.push(["Backing Filesystem", "extfs"]);
    }],
    ["driver-status case", (value: MutableInfoProjection) => {
      value.driverStatus = [["Driver-Type", "io.containerd.snapshotter.v1"]];
    }],
    ["engine version", (value: MutableInfoProjection) => { value.serverVersion = "29.6.0"; }],
    ["engine OS", (value: MutableInfoProjection) => { value.osType = "windows"; }],
    ["engine architecture", (value: MutableInfoProjection) => { value.architecture = "aarch64"; }],
  ])("fails before image inspection on mismatched containerd-store attestation %s", async (_name, mutate) => {
    const { adapter, requests } = createProbeAdapter(prepared, { mutateInfo: mutate });
    await expect(__observeCppCuteAotLocalDockerImageWithProcessForTest(
      prepared.authorized,
      adapter,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-MISMATCH" });
    expect(requests.map(probeFromRequest)).toEqual(["version", "info"]);
    expect(requests[0] === undefined ? true : existsSync(requests[0].cwd)).toBe(false);
  });

  it.each([
    ["manifest digest", (value: MutableProjection) => { value.descriptor.digest = `sha256:${"e".repeat(64)}`; }],
    ["manifest media type", (value: MutableProjection) => { value.descriptor.mediaType = "application/vnd.docker.distribution.manifest.v2+json"; }],
    ["manifest size", (value: MutableProjection) => { value.descriptor.size += 1; }],
    ["containerd image ID", (value: MutableProjection) => { value.id = prepared.fixture.configDigest; }],
    ["platform OS", (value: MutableProjection) => { value.platform.os = "windows"; }],
    ["platform architecture", (value: MutableProjection) => { value.platform.architecture = "arm64"; }],
    ["platform variant", (value: MutableProjection) => { value.platform.variant = "v8"; }],
    ["platform OS version", (value: MutableProjection) => { value.platform.osVersion = "6.1"; }],
    ["repository digest", (value: MutableProjection) => { value.repoDigests = [`ghcr.io/other/image@${prepared.fixture.manifestDigest}`]; }],
    ["rootfs type", (value: MutableProjection) => { value.rootfs.type = "other"; }],
    ["diff-ID order", (value: MutableProjection) => { value.rootfs.diffIds.reverse(); }],
    ["nonempty environment", (value: MutableProjection) => { (value.config as Record<string, unknown>).Env = ["PATH=/escape"]; }],
    ["nonempty labels", (value: MutableProjection) => { (value.config as Record<string, unknown>).Labels = { privileged: "" }; }],
  ])("rejects mismatched %s", async (_name, mutate) => {
    const { adapter, requests } = createProbeAdapter(prepared, { mutateImage: mutate });
    await expect(__observeCppCuteAotLocalDockerImageWithProcessForTest(
      prepared.authorized,
      adapter,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-MISMATCH" });
    expect(requests.map(probeFromRequest)).toEqual(["version", "info", "image"]);
    expect(requests[0] === undefined ? true : existsSync(requests[0].cwd)).toBe(false);
  });

  it("rejects duplicate repository digests", async () => {
    const { adapter, requests } = createProbeAdapter(prepared, {
      mutateImage: (value) => { value.repoDigests.push(value.repoDigests[0]!); },
    });
    await expect(__observeCppCuteAotLocalDockerImageWithProcessForTest(
      prepared.authorized,
      adapter,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-OUTPUT" });
    expect(requests.map(probeFromRequest)).toEqual(DOCKER_PROBE_SEQUENCE);
    expect(requests[0] === undefined ? true : existsSync(requests[0].cwd)).toBe(false);
  });

  it.each(DOCKER_PROBE_SEQUENCE)("rejects closed-schema drift at the %s probe", async (stage) => {
    const options: ProbeAdapterOptions = stage === "version"
      ? { mutateVersion: (value) => { value.ambient = true; } }
      : stage === "info"
        ? { mutateInfo: (value) => { value.ambient = true; } }
        : { mutateImage: (value) => { value.ambient = true; } };
    const { adapter, requests } = createProbeAdapter(prepared, options);
    await expect(__observeCppCuteAotLocalDockerImageWithProcessForTest(
      prepared.authorized,
      adapter,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-OUTPUT" });
    expect(requests.map(probeFromRequest)).toEqual(
      DOCKER_PROBE_SEQUENCE.slice(0, DOCKER_PROBE_SEQUENCE.indexOf(stage) + 1),
    );
    expect(requests[0] === undefined ? true : existsSync(requests[0].cwd)).toBe(false);
  });

  it.each(DOCKER_PROBE_SEQUENCE.flatMap((stage) => [
    [stage, "trailing document", (json: string) => `${json}\n{}\n`],
    [stage, "CRLF", (json: string) => `${json}\r\n`],
    [stage, "missing LF", (json: string) => json],
    [stage, "duplicate key", (json: string) => `${json.replace(/^\{/u, '{"schema":"duplicate",')}\n`],
  ] as const))("strict-decodes the %s probe and rejects %s", async (stage, _name, encode) => {
    const { adapter, requests } = createProbeAdapter(prepared, {
      resultFor: (probe, value) => probe === stage
        ? Object.freeze({
            exitCode: 0,
            signal: null,
            stdout: encoder.encode(encode(JSON.stringify(value))),
            stderr: new Uint8Array(),
          })
        : resultFromProjection(value),
    });
    await expect(__observeCppCuteAotLocalDockerImageWithProcessForTest(
      prepared.authorized,
      adapter,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-OUTPUT" });
    expect(requests.map(probeFromRequest)).toEqual(
      DOCKER_PROBE_SEQUENCE.slice(0, DOCKER_PROBE_SEQUENCE.indexOf(stage) + 1),
    );
    expect(requests[0] === undefined ? true : existsSync(requests[0].cwd)).toBe(false);
  });

  it.each(DOCKER_PROBE_SEQUENCE.flatMap((stage) => [
    [stage, "stdout"],
    [stage, "stderr"],
  ] as const))("rejects the %s probe when snapshotted %s is cap plus one", async (stage, stream) => {
    const { adapter, requests } = createProbeAdapter(prepared, {
      resultFor: (probe, value, request) => {
        const normal = resultFromProjection(value);
        if (probe !== stage) return normal;
        return Object.freeze({
          ...normal,
          stdout: stream === "stdout"
            ? new Uint8Array(request.stdoutByteLimit + 1)
            : normal.stdout,
          stderr: stream === "stderr"
            ? new Uint8Array(request.stderrByteLimit + 1)
            : normal.stderr,
        });
      },
    });
    await expect(__observeCppCuteAotLocalDockerImageWithProcessForTest(
      prepared.authorized,
      adapter,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-OUTPUT" });
    expect(requests.map(probeFromRequest)).toEqual(
      DOCKER_PROBE_SEQUENCE.slice(0, DOCKER_PROBE_SEQUENCE.indexOf(stage) + 1),
    );
    expect(requests[0] === undefined ? true : existsSync(requests[0].cwd)).toBe(false);
  });

  it.each(DOCKER_PROBE_SEQUENCE)("rejects nonzero exit at the %s probe", async (stage) => {
    const { adapter, requests } = createProbeAdapter(prepared, {
      resultFor: (probe, value) => Object.freeze({
        ...resultFromProjection(value),
        exitCode: probe === stage ? 1 : 0,
      }),
    });
    await expect(__observeCppCuteAotLocalDockerImageWithProcessForTest(
      prepared.authorized,
      adapter,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-EXIT" });
    expect(requests.map(probeFromRequest)).toEqual(
      DOCKER_PROBE_SEQUENCE.slice(0, DOCKER_PROBE_SEQUENCE.indexOf(stage) + 1),
    );
    expect(requests[0] === undefined ? true : existsSync(requests[0].cwd)).toBe(false);
  });

  it.each(DOCKER_PROBE_SEQUENCE)("rejects successful stderr at the %s probe", async (stage) => {
    const { adapter, requests } = createProbeAdapter(prepared, {
      resultFor: (probe, value) => Object.freeze({
        ...resultFromProjection(value),
        stderr: probe === stage ? encoder.encode("warning") : new Uint8Array(),
      }),
    });
    await expect(__observeCppCuteAotLocalDockerImageWithProcessForTest(
      prepared.authorized,
      adapter,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-OUTPUT" });
    expect(requests.map(probeFromRequest)).toEqual(
      DOCKER_PROBE_SEQUENCE.slice(0, DOCKER_PROBE_SEQUENCE.indexOf(stage) + 1),
    );
    expect(requests[0] === undefined ? true : existsSync(requests[0].cwd)).toBe(false);
  });

  it("rejects hostile result shape and unknown adapter failures without leaking details", async () => {
    let hostileCwd: string | undefined;
    await expect(__observeCppCuteAotLocalDockerImageWithProcessForTest(
      prepared.authorized,
      async (request) => {
        hostileCwd = request.cwd;
        const hostile: Record<string, unknown> = {
          exitCode: 0,
          signal: null,
          stderr: new Uint8Array(),
        };
        Object.defineProperty(hostile, "stdout", {
          enumerable: true,
          get: () => encoder.encode("{}\n"),
        });
        return hostile as unknown as BoundedChildProcessResult;
      },
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-OUTPUT" });
    expect(hostileCwd === undefined ? true : existsSync(hostileCwd)).toBe(false);

    let failedCwd: string | undefined;
    await expect(__observeCppCuteAotLocalDockerImageWithProcessForTest(
      prepared.authorized,
      async (request) => {
        failedCwd = request.cwd;
        throw new Error("attacker message");
      },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-PROCESS",
      message: expect.not.stringContaining("attacker message"),
    });
    expect(failedCwd === undefined ? true : existsSync(failedCwd)).toBe(false);
  });

  it.each(DOCKER_PROBE_SEQUENCE)("detects and cleans private config mutation after the %s adapter", async (stage) => {
    const { adapter, requests } = createProbeAdapter(prepared, {
      afterProbe: (probe, request) => {
        if (probe === stage) {
          writeFileSync(join(request.cwd, "docker-config", "config.json"), "{}", { mode: 0o600 });
        }
      },
    });
    await expect(__observeCppCuteAotLocalDockerImageWithProcessForTest(
      prepared.authorized,
      adapter,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-OUTPUT" });
    expect(requests.map(probeFromRequest)).toEqual(
      DOCKER_PROBE_SEQUENCE.slice(0, DOCKER_PROBE_SEQUENCE.indexOf(stage) + 1),
    );
    expect(requests[0] === undefined ? true : existsSync(requests[0].cwd)).toBe(false);
  });

  it.each(DOCKER_PROBE_SEQUENCE)("detects and cleans private HOME mutation after the %s adapter", async (stage) => {
    const { adapter, requests } = createProbeAdapter(prepared, {
      afterProbe: (probe, request) => {
        if (probe === stage) {
          writeFileSync(join(request.cwd, "home", ".docker-state"), "forbidden", { mode: 0o600 });
        }
      },
    });
    await expect(__observeCppCuteAotLocalDockerImageWithProcessForTest(
      prepared.authorized,
      adapter,
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-OUTPUT" });
    expect(requests.map(probeFromRequest)).toEqual(
      DOCKER_PROBE_SEQUENCE.slice(0, DOCKER_PROBE_SEQUENCE.indexOf(stage) + 1),
    );
    expect(requests[0] === undefined ? true : existsSync(requests[0].cwd)).toBe(false);
  });

  it.each(DOCKER_PROBE_SEQUENCE)("honors cancellation immediately after the %s adapter and cleans staging", async (stage) => {
    const controller = new AbortController();
    const { adapter, requests } = createProbeAdapter(prepared, {
      afterProbe: (probe) => {
        if (probe === stage) controller.abort(new Error("must not leak"));
      },
    });
    await expect(__observeCppCuteAotLocalDockerImageWithProcessForTest(
      prepared.authorized,
      adapter,
      { signal: controller.signal },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-CANCELLED",
      message: expect.not.stringContaining("must not leak"),
    });
    expect(requests.map(probeFromRequest)).toEqual(
      DOCKER_PROBE_SEQUENCE.slice(0, DOCKER_PROBE_SEQUENCE.indexOf(stage) + 1),
    );
    expect(requests[0] === undefined ? true : existsSync(requests[0].cwd)).toBe(false);
  });

  it("pre-aborts before filesystem or process effects", async () => {
    const before = new Set(readdirSync("/tmp").filter((entry) => entry.startsWith("browsergrad-cpp-cute-docker-")));
    const adapter = vi.fn(async () => resultFromProjection(projection(prepared)));
    const controller = new AbortController();
    controller.abort(new Error("must not leak"));
    await expect(__observeCppCuteAotLocalDockerImageWithProcessForTest(
      prepared.authorized,
      adapter,
      { signal: controller.signal },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-DOCKER-IMAGE-CANCELLED",
      message: expect.not.stringContaining("must not leak"),
    });
    expect(adapter).not.toHaveBeenCalled();
    const after = new Set(readdirSync("/tmp").filter((entry) => entry.startsWith("browsergrad-cpp-cute-docker-")));
    expect(after).toEqual(before);
  });
});
