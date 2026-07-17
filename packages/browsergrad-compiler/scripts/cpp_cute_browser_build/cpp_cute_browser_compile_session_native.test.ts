import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import {
  canonicalJsonBytes,
  compareCanonicalStrings,
  encodeWireU64,
  sha256Hex,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { describe, expect, it } from "vitest";

import { assembleCppCuteBrowserInputFrameRegions } from "../../src/cpp_cute_browser_input_frame.js";
import {
  deriveCppCuteFrontendEntryRequestId,
  deriveCppCuteFrontendRequestHash,
  deriveCppCuteFrontendSourceFileId,
  prepareCppCuteFrontendRequest,
  unwrapPreparedCppCuteFrontendRequest,
  type CppCuteFrontendEntryRequestV1,
  type CppCuteFrontendRequestBodyV1,
  type CppCuteFrontendRequestLimitsV1,
  type CppCuteFrontendRequestSourceFileV1,
  type CppCuteFrontendRequestV1,
  type CppCuteFrontendSourceSnapshotInput,
} from "../../src/cpp_cute_frontend_request.js";
import {
  prepareCppCuteFrontendProfile,
  unwrapPreparedCppCuteFrontendProfile,
  type CppCuteFrontendExtractionLimits,
  type CppCuteFrontendProfileV2,
} from "../../src/cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
  CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
} from "../../src/cpp_cute_frontend_types.js";
import {
  createCppCuteBrowserProfileInput,
  createCppCuteProfileInput,
} from "../../tests/compiler/support/cpp_cute_frontend_fixtures.js";

const scriptRoot = dirname(fileURLToPath(import.meta.url));
const extractorRoot = join(scriptRoot, "extractor");
const nativeSource = join(scriptRoot, "cpp_cute_browser_compile_session_native_test.cpp");
const compiler = ["/usr/bin/clang++", "/usr/bin/c++", "/usr/bin/g++"]
  .find((candidate) => existsSync(candidate));
const encoder = new TextEncoder();
const mainPath = "/workspace/src/main.cu";
const headerPath = "/workspace/src/project.hpp";
const mainBytes = encoder.encode('#include "project.hpp"\nauto layout = make_layout(Int<2>{});\n');
const headerBytes = encoder.encode("constexpr int project_value = 2;\n");

interface GoldenFixture {
  readonly profile: CppCuteFrontendProfileV2;
  readonly request: CppCuteFrontendRequestV1;
  readonly snapshots: readonly CppCuteFrontendSourceSnapshotInput[];
  readonly profileHash: string;
  readonly compilationContractHash: string;
  readonly requestHash: string;
}

async function goldenFixture(): Promise<GoldenFixture> {
  const profileInput = structuredClone(createCppCuteBrowserProfileInput()) as unknown as Record<string, unknown>;
  const language = profileInput["language"] as Record<string, unknown>;
  const options = language["options"] as unknown[];
  options.push(
    { kind: "undefine", name: "NDEBUG" },
    { kind: "warning-policy", id: "clang.unused-variable", disposition: "error" },
  );
  const preparedProfile = await prepareCppCuteFrontendProfile(profileInput);
  const profile = unwrapPreparedCppCuteFrontendProfile(preparedProfile).profile;
  const files = await Promise.all([
    sourceFile("main-source", mainPath, null, mainBytes),
    sourceFile("project-header", headerPath, "workspace-source", headerBytes),
  ]);
  files.sort((left, right) => compareCanonicalStrings(left.virtualPath, right.virtualPath));
  const tokenBegin = new TextDecoder().decode(mainBytes).indexOf("layout");
  const anchor = {
    virtualPath: mainPath,
    beginByte: encodeWireU64(BigInt(tokenBegin)),
    endByte: encodeWireU64(BigInt(tokenBegin + "layout".length)),
    tokenSha256: await sha256Hex(mainBytes.subarray(tokenBegin, tokenBegin + "layout".length)),
  };
  const entryBody = {
    requestId: `bg.cpp.entry-request.sha256.${"0".repeat(64)}`,
    kind: "layout" as const,
    declarationKind: "variable" as const,
    anchor,
  };
  const entry: CppCuteFrontendEntryRequestV1 = {
    ...entryBody,
    requestId: await deriveCppCuteFrontendEntryRequestId(entryBody),
  };
  const limits = semanticLimits(preparedProfile.extractionLimits);
  const body: CppCuteFrontendRequestBodyV1 = {
    schema: "browsergrad.compiler.cpp-cute.frontend-request",
    version: { major: 1, minor: 0 },
    compilationContractHash: preparedProfile.compilationContractHash,
    mainVirtualPath: mainPath,
    files,
    entryRequests: [entry],
    expectedArtifact: {
      schema: "browsergrad.compiler.cpp-cute.frontend-artifact",
      version: {
        major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
        minor: CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
      },
    },
    limits,
  };
  const request: CppCuteFrontendRequestV1 = {
    ...body,
    requestId: `bg.cpp.frontend-request.sha256.${await deriveCppCuteFrontendRequestHash(body)}`,
  };
  const snapshots = files.map((file) => ({
    virtualPath: file.virtualPath,
    bytes: file.role === "main-source" ? mainBytes : headerBytes,
  }));
  const preparedRequest = await prepareCppCuteFrontendRequest(
    preparedProfile,
    request,
    snapshots,
  );
  const verifiedRequest = unwrapPreparedCppCuteFrontendRequest(preparedRequest).request;
  return {
    profile,
    request: verifiedRequest,
    snapshots,
    profileHash: preparedProfile.profileHash,
    compilationContractHash: preparedProfile.compilationContractHash,
    requestHash: preparedRequest.requestHash,
  };
}

async function sourceFile(
  role: CppCuteFrontendRequestSourceFileV1["role"],
  virtualPath: string,
  includeRootId: string | null,
  bytes: Uint8Array,
): Promise<CppCuteFrontendRequestSourceFileV1> {
  const body = {
    role,
    virtualPath,
    contentSha256: await sha256Hex(bytes),
    byteLength: encodeWireU64(BigInt(bytes.byteLength)),
    includeRootId,
  };
  return { ...body, fileId: await deriveCppCuteFrontendSourceFileId(body) };
}

function semanticLimits(
  limits: CppCuteFrontendExtractionLimits,
): CppCuteFrontendRequestLimitsV1 {
  const keys = [
    "maxSourceFiles", "maxSourceBytes", "maxHeaderFiles", "maxHeaderBytes",
    "maxIncludeDepth", "maxMacroExpansions", "maxPreprocessedTokens",
    "maxAstNodes", "maxConstexprSteps", "maxTemplateInstantiations",
    "maxTemplateDepth", "maxDeclarations", "maxTypes", "maxConstants",
    "maxLayouts", "maxTensors", "maxOperations", "maxTargetIntrinsics",
    "maxDiagnostics", "maxOutputBytes",
  ] as const;
  return Object.fromEntries(keys.map((key) => [key, limits[key]])) as CppCuteFrontendRequestLimitsV1;
}

function frame(
  profile: unknown,
  request: unknown,
  snapshots: readonly CppCuteFrontendSourceSnapshotInput[],
  rawProfile?: Uint8Array,
): Uint8Array {
  return assembleCppCuteBrowserInputFrameRegions({
    profileRegionBytes: rawProfile ?? canonicalJsonBytes(profile),
    requestRegionBytes: canonicalJsonBytes(request),
    sourceSnapshots: snapshots,
    limits: {
      maxFrameByteLength: 4 * 1024 * 1024,
      maxSourceSnapshotCount: snapshots.length,
      maxSourceSnapshotByteLength: mainBytes.byteLength + headerBytes.byteLength,
    },
  }).frameBytes;
}

async function cases(): Promise<readonly {
  readonly name: string;
  readonly status: "ready" | "invalid" | "abi" | "resource";
  readonly bytes: Uint8Array;
}[]> {
  const fixture = await goldenFixture();
  const futureProfile = structuredClone(fixture.profile) as unknown as Record<string, unknown>;
  (futureProfile["version"] as Record<string, unknown>)["minor"] = 99;
  const oversizedOptions = structuredClone(fixture.profile) as unknown as Record<string, unknown>;
  (oversizedOptions["language"] as Record<string, unknown>)["options"] =
    Array.from({ length: 4_097 }, (_, index) => ({
      kind: "define",
      name: `BROWSERGRAD_OPTION_${index}`,
      value: null,
    }));
  const identityDrift = structuredClone(fixture.request) as unknown as Record<string, unknown>;
  const driftFiles = identityDrift["files"] as Record<string, unknown>[];
  driftFiles[0]!["contentSha256"] = "f".repeat(64);
  const widened = structuredClone(fixture.request) as unknown as Record<string, unknown>;
  (widened["limits"] as Record<string, unknown>)["maxOutputBytes"] = 8 * 1024 * 1024 + 1;
  const contractDrift = structuredClone(fixture.request) as unknown as Record<string, unknown>;
  contractDrift["compilationContractHash"] = "0".repeat(64);
  const compilerDrift = structuredClone(fixture.profile) as unknown as Record<string, unknown>;
  (((compilerDrift["toolchain"] as Record<string, unknown>)["compiler"] as Record<string, unknown>))["version"] =
    "22.1.9";
  const runtimeDrift = structuredClone(fixture.profile) as unknown as Record<string, unknown>;
  const runtimeDeployment = runtimeDrift["deployment"] as Record<string, unknown>;
  ((runtimeDeployment["compilerRuntime"] as Record<string, unknown>))["runtimeAbiManifestSha256"] =
    "0".repeat(64);
  const diagnosticDrift = structuredClone(fixture.profile) as unknown as Record<string, unknown>;
  const diagnosticLanguage = diagnosticDrift["language"] as Record<string, unknown>;
  ((diagnosticLanguage["diagnostics"] as Record<string, unknown>))["normalizationManifestSha256"] =
    "0".repeat(64);
  const unknown = structuredClone(fixture.request) as unknown as Record<string, unknown>;
  unknown["argv"] = ["clang++"];
  const profileText = new TextDecoder().decode(canonicalJsonBytes(fixture.profile));
  return [
    { name: "ready", status: "ready", bytes: frame(fixture.profile, fixture.request, fixture.snapshots) },
    { name: "noncanonical", status: "invalid", bytes: frame(
      fixture.profile,
      fixture.request,
      fixture.snapshots,
      encoder.encode(` ${profileText}`),
    ) },
    { name: "future-version", status: "abi", bytes: frame(futureProfile, fixture.request, fixture.snapshots) },
    { name: "aot-deployment", status: "abi", bytes: frame(createCppCuteProfileInput(), fixture.request, fixture.snapshots) },
    { name: "option-budget", status: "resource", bytes: frame(oversizedOptions, fixture.request, fixture.snapshots) },
    { name: "source-identity", status: "invalid", bytes: frame(fixture.profile, identityDrift, fixture.snapshots) },
    { name: "widened-limit", status: "resource", bytes: frame(fixture.profile, widened, fixture.snapshots) },
    { name: "contract-drift", status: "invalid", bytes: frame(fixture.profile, contractDrift, fixture.snapshots) },
    { name: "compiler-drift", status: "abi", bytes: frame(compilerDrift, fixture.request, fixture.snapshots) },
    { name: "runtime-drift", status: "abi", bytes: frame(runtimeDrift, fixture.request, fixture.snapshots) },
    { name: "diagnostic-drift", status: "abi", bytes: frame(diagnosticDrift, fixture.request, fixture.snapshots) },
    { name: "closed-schema", status: "invalid", bytes: frame(fixture.profile, unknown, fixture.snapshots) },
  ];
}

async function compileAndRun(extraFlags: readonly string[]): Promise<void> {
  if (compiler === undefined) throw new Error("native C++ compiler unavailable");
  const fixture = await goldenFixture();
  const testCases = await cases();
  const workingDirectory = mkdtempSync(join(tmpdir(), "browsergrad-compile-session-"));
  const executable = join(workingDirectory, "compile-session-native-test");
  try {
    const compilation = spawnSync(compiler, [
      "-std=c++20", "-O1", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
      "-fno-omit-frame-pointer", ...extraFlags,
      nativeSource,
      join(extractorRoot, "BrowserGradCppCuteCompileSession.cpp"),
      join(extractorRoot, "BrowserGradCppCuteCanonicalJson.cpp"),
      join(extractorRoot, "BrowserGradCppCuteSha256.cpp"),
      "-I", scriptRoot,
      "-o", executable,
    ], { encoding: "utf8", timeout: 60_000 });
    expect(compilation.error).toBeUndefined();
    expect(compilation.status, compilation.stderr).toBe(0);

    for (const testCase of testCases) {
      const fixturePath = join(workingDirectory, `${testCase.name}.frame`);
      writeFileSync(fixturePath, testCase.bytes);
      const execution = spawnSync(executable, [
        fixturePath,
        testCase.status,
        fixture.profileHash,
        fixture.compilationContractHash,
        fixture.requestHash,
      ], {
        encoding: "utf8",
        timeout: 30_000,
        env: {
          ...process.env,
          ASAN_OPTIONS: "detect_leaks=1:halt_on_error=1",
          UBSAN_OPTIONS: "halt_on_error=1:print_stacktrace=1",
        },
      });
      expect(execution.error).toBeUndefined();
      expect(execution.status,
        `${testCase.name}: signal=${execution.signal ?? "none"}\n${execution.stdout}\n${execution.stderr}`,
      ).toBe(0);
    }
  } finally {
    rmSync(workingDirectory, { recursive: true, force: true });
  }
}

describe("bounded native C++/CuTe compile-session admission", () => {
  it.skipIf(compiler === undefined)(
    "admits exact TS identities and rejects hostile, drifted, and over-budget frames",
    () => compileAndRun([]),
    90_000,
  );

  it.skipIf(compiler !== "/usr/bin/clang++")(
    "stays clean under undefined-behavior sanitizer coverage",
    () => compileAndRun(["-fsanitize=undefined"]),
    90_000,
  );

  // Apple clang's ASan runtime deadlocks during dyld initialization on the
  // Darwin runner; Linux CI owns address/leak sanitizer coverage.
  it.skipIf(compiler !== "/usr/bin/clang++" || process.platform === "darwin")(
    "stays clean under address and leak sanitizer coverage",
    () => compileAndRun(["-fsanitize=address"]),
    90_000,
  );
});
