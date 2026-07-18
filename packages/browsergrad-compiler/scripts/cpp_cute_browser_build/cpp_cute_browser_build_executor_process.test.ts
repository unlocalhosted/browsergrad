import { lstat, mkdtemp, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  CPP_CUTE_BROWSER_BUILD_EXECUTOR_PROCESS,
  CppCuteBrowserBuildProcessError,
} from "./cpp_cute_browser_build_executor_process.mjs";

const temporaryRoots: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map((root) => rm(root, {
    recursive: true,
    force: true,
  })));
});

async function temporaryRoot(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "browsergrad-build-process-"));
  temporaryRoots.push(root);
  return root;
}

function input(
  cwd: string,
  args: readonly string[],
  overrides: Partial<{
    maximumOutputByteLength: number;
    maximumDurationMs: number;
    signal: AbortSignal;
  }> = {},
) {
  return {
    executable: process.execPath,
    arguments: args,
    cwd,
    environment: { BROWSERGRAD_EXACT_ENV: "present" },
    maximumOutputByteLength: 16 * 1024,
    maximumDurationMs: 10_000,
    ...overrides,
  };
}

describe("Clang-Wasm build process boundary", () => {
  it("executes exact argv, cwd, and environment without a shell", async () => {
    const root = await temporaryRoot();
    const shellMarker = join(root, "must-not-exist");
    const hostileArgument = `$(touch ${shellMarker})`;
    const result = await CPP_CUTE_BROWSER_BUILD_EXECUTOR_PROCESS.run(input(root, [
      "-e",
      "process.stdout.write(JSON.stringify({argv:process.argv.slice(1),cwd:process.cwd(),exact:process.env.BROWSERGRAD_EXACT_ENV,path:process.env.PATH}))",
      hostileArgument,
    ]));

    expect(result).toMatchObject({ exitCode: 0, terminationSignal: null });
    expect(JSON.parse(Buffer.from(result.stdout).toString("utf8"))).toEqual({
      argv: [hostileArgument],
      cwd: await realpath(root),
      exact: "present",
    });
    expect(Buffer.from(result.stderr).toString("utf8")).toBe("");
    await expect(lstat(shellMarker)).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("returns nonzero exits with their bounded diagnostic bytes", async () => {
    const root = await temporaryRoot();
    const result = await CPP_CUTE_BROWSER_BUILD_EXECUTOR_PROCESS.run(input(root, [
      "-e",
      "process.stdout.write('out');process.stderr.write('err');process.exitCode=7",
    ]));

    expect(result.exitCode).toBe(7);
    expect(result.terminationSignal).toBeNull();
    expect(Buffer.from(result.stdout).toString("utf8")).toBe("out");
    expect(Buffer.from(result.stderr).toString("utf8")).toBe("err");
    expect(result.stdoutByteLength).toBe(3);
    expect(result.stderrByteLength).toBe(3);
  });

  it("streams bounded output with backpressure instead of retaining duplicate bytes", async () => {
    const root = await temporaryRoot();
    const chunks: Array<Readonly<{ stream: "stdout" | "stderr"; text: string }>> = [];
    const result = await CPP_CUTE_BROWSER_BUILD_EXECUTOR_PROCESS.run({
      ...input(root, [
        "-e",
        "process.stdout.write('streamed-out');process.stderr.write('streamed-err')",
      ]),
      onOutputChunk: async (stream, chunk) => {
        await Promise.resolve();
        chunks.push({ stream, text: Buffer.from(chunk).toString("utf8") });
      },
    });

    expect(result.stdout).toHaveLength(0);
    expect(result.stderr).toHaveLength(0);
    expect(result.stdoutByteLength).toBe("streamed-out".length);
    expect(result.stderrByteLength).toBe("streamed-err".length);
    expect(chunks.filter(({ stream }) => stream === "stdout").map(({ text }) => text).join(""))
      .toBe("streamed-out");
    expect(chunks.filter(({ stream }) => stream === "stderr").map(({ text }) => text).join(""))
      .toBe("streamed-err");
  });

  it("persists the exact admitted prefix before output-limit and sink failures", async () => {
    const outputRoot = await temporaryRoot();
    const admitted: Uint8Array[] = [];
    await expect(CPP_CUTE_BROWSER_BUILD_EXECUTOR_PROCESS.run({
      ...input(outputRoot, ["-e", "process.stdout.write('x'.repeat(2048))"], {
        maximumOutputByteLength: 1024,
      }),
      onOutputChunk: async (_stream, chunk) => {
        admitted.push(new Uint8Array(chunk));
      },
    })).rejects.toMatchObject({ reason: "output-limit" });
    expect(admitted.reduce((total, chunk) => total + chunk.byteLength, 0)).toBe(1024);

    const sinkRoot = await temporaryRoot();
    await expect(CPP_CUTE_BROWSER_BUILD_EXECUTOR_PROCESS.run({
      ...input(sinkRoot, ["-e", "process.stdout.write('output')"]),
      onOutputChunk: async () => {
        throw new Error("injected sink failure");
      },
    })).rejects.toMatchObject({
      reason: "output-sink",
      cause: expect.objectContaining({ message: "injected sink failure" }),
    });
  });

  it("terminates process groups when output, duration, or cancellation bounds fire", async () => {
    const outputRoot = await temporaryRoot();
    await expect(CPP_CUTE_BROWSER_BUILD_EXECUTOR_PROCESS.run(input(outputRoot, [
      "-e",
      "process.stdout.write('x'.repeat(1025))",
    ], { maximumOutputByteLength: 1024 }))).rejects.toMatchObject({
      reason: "output-limit",
    });

    const timeoutRoot = await temporaryRoot();
    await expect(CPP_CUTE_BROWSER_BUILD_EXECUTOR_PROCESS.run(input(timeoutRoot, [
      "-e",
      "setInterval(()=>{},1000)",
    ], { maximumDurationMs: 25 }))).rejects.toMatchObject({
      reason: "timeout",
    });

    const cancelledRoot = await temporaryRoot();
    const controller = new AbortController();
    const pending = CPP_CUTE_BROWSER_BUILD_EXECUTOR_PROCESS.run(input(cancelledRoot, [
      "-e",
      "setInterval(()=>{},1000)",
    ], { signal: controller.signal }));
    controller.abort();
    await expect(pending).rejects.toMatchObject({ reason: "cancelled" });
  });

  it("reports spawn failures through its typed boundary", async () => {
    const root = await temporaryRoot();
    await expect(CPP_CUTE_BROWSER_BUILD_EXECUTOR_PROCESS.run({
      ...input(root, []),
      executable: join(root, "missing-executable"),
    })).rejects.toMatchObject({
      name: "CppCuteBrowserBuildProcessError",
      reason: "spawn",
    });
    expect(new CppCuteBrowserBuildProcessError("spawn", "test")).toBeInstanceOf(Error);
  });
});
