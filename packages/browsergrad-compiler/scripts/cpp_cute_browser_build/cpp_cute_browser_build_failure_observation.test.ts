import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { afterEach, describe, expect, it } from "vitest";

import {
  persistCppCuteBrowserBuildFailureObservation,
} from "./cpp_cute_browser_build_failure_observation.mjs";

const roots: string[] = [];
const lockId =
  "bg.cpp.browser-build-input-lock.sha256.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const sourceSetSha256 =
  "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, {
    recursive: true,
    force: true,
  })));
});

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "browsergrad-build-failure-"));
  roots.push(root);
  const outputRoot = join(root, "output");
  const stateRoot = join(root, "state");
  const logRoot = join(stateRoot, "evidence", "build-logs");
  await Promise.all([
    mkdir(outputRoot, { mode: 0o700 }),
    mkdir(logRoot, { recursive: true, mode: 0o700 }),
  ]);
  return { root, outputRoot, stateRoot, logRoot };
}

describe("Clang-Wasm build failure observation", () => {
  it("binds a typed failure to exact immutable partial-log bytes", async () => {
    const input = await fixture();
    await Promise.all([
      writeFile(join(input.logRoot, "native-tablegen-configure.stdout.log"), "configured\n"),
      writeFile(join(input.logRoot, "native-tablegen-configure.stderr.log"), ""),
    ]);
    const cause = Object.assign(new Error("configure rejected target flags"), {
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: "$.configuredTarget",
    });
    const result = await persistCppCuteBrowserBuildFailureObservation({
      outputRoot: input.outputRoot,
      stateRoot: input.stateRoot,
      lockId,
      sourceSetSha256,
      cause,
    });
    expect(result).toMatchObject({
      partialLogCount: 2,
      successfulBuildReceiptWritten: false,
      sha256: expect.stringMatching(/^[0-9a-f]{64}$/u),
    });
    const bytes = await readFile(result.outputPath);
    const observation = JSON.parse(new TextDecoder().decode(bytes)) as {
      authority: string;
      failure: { code: string; path: string };
      partialLogs: Array<{ sha256: string; byteLength: number }>;
      claims: { releaseReady: boolean };
    };
    expect(Uint8Array.from(bytes)).toEqual(canonicalJsonBytes(observation));
    expect(observation).toMatchObject({
      authority: "build-failure-observation-only",
      failure: { code: cause.code, path: cause.path },
      claims: { releaseReady: false },
    });
    expect(observation.partialLogs).toHaveLength(2);
    expect(observation.partialLogs[0]?.sha256).toMatch(/^[0-9a-f]{64}$/u);
  });

  it("rejects symlinked partial logs and an existing receipt", async () => {
    const symlinked = await fixture();
    const outside = join(symlinked.root, "outside.log");
    await writeFile(outside, "outside\n");
    await symlink(
      outside,
      join(symlinked.logRoot, "native-tablegen-configure.stdout.log"),
    );
    await expect(persistCppCuteBrowserBuildFailureObservation({
      outputRoot: symlinked.outputRoot,
      stateRoot: symlinked.stateRoot,
      lockId,
      sourceSetSha256,
      cause: new Error("failed"),
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-FAILURE-OBSERVATION-INVALID",
    });

    const existing = await fixture();
    const request = {
      outputRoot: existing.outputRoot,
      stateRoot: existing.stateRoot,
      lockId,
      sourceSetSha256,
      cause: new Error("failed"),
    };
    await persistCppCuteBrowserBuildFailureObservation(request);
    await expect(persistCppCuteBrowserBuildFailureObservation(request)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-FAILURE-OBSERVATION-CONFLICT",
    });
  });
});
