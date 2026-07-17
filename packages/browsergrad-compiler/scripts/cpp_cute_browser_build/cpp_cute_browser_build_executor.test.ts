import {
  chmod,
  copyFile,
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rename,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

const executorFsFaults = vi.hoisted(() => ({
  afterLink: undefined as
    | ((sourcePath: string, destinationPath: string) => Promise<void>)
    | undefined,
  syncDirectoryFailure: undefined as
    | { path: string; remaining: number; error: Error }
    | undefined,
  closeTemporaryFailure: undefined as Error | undefined,
}));

vi.mock("./cpp_cute_browser_build_executor_fs.mjs", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("./cpp_cute_browser_build_executor_fs.mjs")
  >();
  const base = actual.CPP_CUTE_BROWSER_BUILD_EXECUTOR_FS;
  return {
    CPP_CUTE_BROWSER_BUILD_EXECUTOR_FS: Object.freeze({
      ...base,
      link: async (sourcePath: string, destinationPath: string) => {
        await base.link(sourcePath, destinationPath);
        const afterLink = executorFsFaults.afterLink;
        executorFsFaults.afterLink = undefined;
        await afterLink?.(sourcePath, destinationPath);
      },
      syncDirectory: async (path: string) => {
        const failure = executorFsFaults.syncDirectoryFailure;
        if (failure !== undefined && failure.path === path && failure.remaining > 0) {
          failure.remaining -= 1;
          throw failure.error;
        }
        await base.syncDirectory(path);
      },
      closeFileHandle: async (
        handle: import("node:fs/promises").FileHandle,
        purpose: string,
      ) => {
        await base.closeFileHandle(handle, purpose);
        if (purpose === "temporary-sidecar" &&
            executorFsFaults.closeTemporaryFailure !== undefined) {
          const failure = executorFsFaults.closeTemporaryFailure;
          executorFsFaults.closeTemporaryFailure = undefined;
          throw failure;
        }
      },
    }),
  };
});

import {
  cppCuteBrowserBuildInputLockResourceBytes,
  decodeCppCuteBrowserBuildInputLock,
  type PreparedCppCuteBrowserBuildInputLock,
  unwrapPreparedCppCuteBrowserBuildInputLock,
} from "../../dist/cpp_cute_browser_build_lock.js";
import {
  CppCuteBrowserBuildExecutorError,
  materializeCppCuteClangWasmSidecar,
  prepareCppCuteClangWasmBuildSource,
  type PrepareCppCuteClangWasmBuildSourceInput,
} from "./cpp_cute_browser_build_executor.mjs";

const checkedInExtractorRoot = join(
  dirname(fileURLToPath(import.meta.url)),
  "extractor",
);
const temporaryRoots: string[] = [];
const MINIMAL_WASM = Uint8Array.of(0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00);

let lock: PreparedCppCuteBrowserBuildInputLock;

beforeAll(async () => {
  lock = await decodeCppCuteBrowserBuildInputLock(
    cppCuteBrowserBuildInputLockResourceBytes(),
  );
});

afterEach(async () => {
  executorFsFaults.afterLink = undefined;
  executorFsFaults.syncDirectoryFailure = undefined;
  executorFsFaults.closeTemporaryFailure = undefined;
  await Promise.all(temporaryRoots.splice(0).map(async (root) => {
    await chmod(join(root, "staged-extractor-source"), 0o700).catch(() => undefined);
    await rm(root, { recursive: true, force: true });
  }));
});

async function fixture(): Promise<{
  readonly root: string;
  readonly input: PrepareCppCuteClangWasmBuildSourceInput;
}> {
  const root = await mkdtemp(join(tmpdir(), "browsergrad-clang-wasm-executor-"));
  temporaryRoots.push(root);
  const sourceRoot = join(root, "source-input");
  const outputRoot = join(root, "output");
  await Promise.all([mkdir(sourceRoot), mkdir(outputRoot)]);
  const sourcePaths = unwrapPreparedCppCuteBrowserBuildInputLock(lock)
    .lock.body.recipe.extractorSource.files.map((file) => file.path);
  await Promise.all(sourcePaths.map((path) => copyFile(
    join(checkedInExtractorRoot, path),
    join(sourceRoot, path),
  )));
  return {
    root,
    input: {
      lock,
      tools: {
        cmakeExecutable: "/tools/cmake/bin/cmake",
        buildToolExecutable: "/usr/bin/make",
        emsdkRoot: "/tools/emsdk",
        emscriptenToolchainFile:
          "/tools/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake",
        emscriptenConfigFile: "/tools/emsdk/.emscripten",
        searchPath: [
          "/tools/cmake/bin",
          "/tools/emsdk/upstream/bin",
          "/tools/emsdk/upstream/emscripten",
          "/tools/python/bin",
          "/tools/node/bin",
        ],
      },
      roots: {
        llvmProjectSourceRoot: join(root, "llvm-source"),
        extractorSourceRoot: join(root, "staged-extractor-source"),
        nativeBuildRoot: join(root, "native-build"),
        wasmBuildRoot: join(root, "wasm-build"),
        outputRoot,
        stateRoot: join(root, "state"),
      },
      extractorSourceInputRoot: sourceRoot,
    },
  };
}

async function writeGeneratedSidecar(
  input: PrepareCppCuteClangWasmBuildSourceInput,
  bytes: Uint8Array = MINIMAL_WASM,
): Promise<void> {
  const generated = join(input.roots.stateRoot, "evidence", "generated");
  await mkdir(generated, { recursive: true });
  await Promise.all([
    writeFile(join(generated, "clang-extractor.wasm"), bytes),
    writeFile(join(generated, "clang-extractor.mjs"), "export default 'factory';\n"),
  ]);
}

function destination(input: PrepareCppCuteClangWasmBuildSourceInput): string {
  return join(input.roots.outputRoot, "browsergrad-cpp-cute", "clang-extractor.wasm");
}

describe("bounded Clang-WASM build source executor", () => {
  it("snapshots the exact locked source set into an owned read-only staging root", async () => {
    const { input } = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(input);

    expect(prepared).toMatchObject({
      authority: "build-source-snapshot-only",
      lockId: lock.lockId,
      sourceSetSha256: lock.extractorSourceSetSha256,
      fileCount: 12,
      sourceVerified: true,
      buildExecuted: false,
      outputIdentityAuthorized: false,
      reproducibilityVerified: false,
      releaseReady: false,
    });
    expect((await readdir(input.roots.extractorSourceRoot)).sort()).toEqual([
      "BrowserGradCppCuteArtifactV3.cpp",
      "BrowserGradCppCuteArtifactV3.h",
      "BrowserGradCppCuteClangAction.cpp",
      "BrowserGradCppCuteClangAction.h",
      "BrowserGradCppCuteExtractor.cpp",
      "BrowserGradCppCuteImportedVfs.cpp",
      "BrowserGradCppCuteImportedVfs.h",
      "BrowserGradCppCuteMetrics.cpp",
      "BrowserGradCppCuteMetrics.h",
      "BrowserGradCppCuteRuntime.cpp",
      "BrowserGradCppCuteRuntime.h",
      "CMakeLists.txt",
    ]);
    const stagedCpp = join(input.roots.extractorSourceRoot, "BrowserGradCppCuteExtractor.cpp");
    const before = await readFile(stagedCpp);
    await writeFile(join(input.extractorSourceInputRoot, "BrowserGradCppCuteExtractor.cpp"), "mutated after snapshot");
    expect(await readFile(stagedCpp)).toEqual(before);
    expect((await lstat(stagedCpp)).mode & 0o222).toBe(0);
  });

  it.each([
    ["extra entry", async (input: PrepareCppCuteClangWasmBuildSourceInput) => {
      await writeFile(join(input.extractorSourceInputRoot, "extra.cpp"), "extra");
    }, "extractorSourceInputRoot"],
    ["missing entry", async (input: PrepareCppCuteClangWasmBuildSourceInput) => {
      await rm(join(input.extractorSourceInputRoot, "CMakeLists.txt"));
    }, "extractorSourceInputRoot"],
    ["symlink entry", async (input: PrepareCppCuteClangWasmBuildSourceInput) => {
      const path = join(input.extractorSourceInputRoot, "CMakeLists.txt");
      await rm(path);
      await symlink(join(checkedInExtractorRoot, "CMakeLists.txt"), path);
    }, "CMakeLists.txt"],
  ])("rejects %s without leaving staged bytes", async (_name, mutate, expectedPath) => {
    const { input } = await fixture();
    await mutate(input);

    await expect(prepareCppCuteClangWasmBuildSource(input)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: expect.stringContaining(expectedPath),
    });
    await expect(lstat(input.roots.extractorSourceRoot)).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("rejects wrong length and same-length wrong content before staging", async () => {
    const first = await fixture();
    await writeFile(join(first.input.extractorSourceInputRoot, "CMakeLists.txt"), "short");
    await expect(prepareCppCuteClangWasmBuildSource(first.input)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-HASH-MISMATCH",
      path: expect.stringContaining("byteLength"),
    });

    const second = await fixture();
    const cppPath = join(second.input.extractorSourceInputRoot, "BrowserGradCppCuteExtractor.cpp");
    const cpp = await readFile(cppPath);
    cpp[0] = cpp[0] === 0 ? 1 : 0;
    await writeFile(cppPath, cpp);
    await expect(prepareCppCuteClangWasmBuildSource(second.input)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-HASH-MISMATCH",
      path: expect.stringContaining("sha256"),
    });
  });

  it("rejects an existing staging root and an already-aborted operation", async () => {
    const conflict = await fixture();
    await mkdir(conflict.input.roots.extractorSourceRoot);
    await expect(prepareCppCuteClangWasmBuildSource(conflict.input)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CONFLICT",
      path: "$.roots.extractorSourceRoot",
    });

    const aborted = await fixture();
    const controller = new AbortController();
    controller.abort();
    await expect(prepareCppCuteClangWasmBuildSource(aborted.input, {
      signal: controller.signal,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CANCELLED",
    });
  });

  it("requires a private staging parent and binds the staged root inode", async () => {
    const unsafeParent = await fixture();
    await chmod(unsafeParent.root, 0o777);
    await expect(prepareCppCuteClangWasmBuildSource(unsafeParent.input)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: "$.roots.extractorSourceRoot.parent",
    });

    const replaced = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(replaced.input);
    await writeGeneratedSidecar(replaced.input);
    const displaced = join(replaced.root, "displaced-staged-source");
    await chmod(replaced.input.roots.extractorSourceRoot, 0o700);
    await rename(replaced.input.roots.extractorSourceRoot, displaced);
    await mkdir(replaced.input.roots.extractorSourceRoot, { mode: 0o700 });
    const sentinel = join(replaced.input.roots.extractorSourceRoot, "sentinel.txt");
    await writeFile(sentinel, "replacement must survive");

    await expect(materializeCppCuteClangWasmSidecar(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CONFLICT",
      path: "$.stagedSourceRoot",
    });
    expect(await readFile(sentinel, "utf8")).toBe("replacement must survive");
  });
});

describe("deterministic Clang-WASM sidecar materializer", () => {
  it("atomically installs exact Wasm bytes, is idempotent, and never distributes the factory", async () => {
    const { input } = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    await writeGeneratedSidecar(input);

    const first = await materializeCppCuteClangWasmSidecar(prepared);
    const second = await materializeCppCuteClangWasmSidecar(prepared);

    expect(second).toEqual(first);
    expect(first).toMatchObject({
      authority: "wasm-sidecar-byte-materialization-observation-only",
      lockId: lock.lockId,
      sourceSetSha256: lock.extractorSourceSetSha256,
      wasmByteLength: MINIMAL_WASM.byteLength,
      distributedWasmPath: destination(input),
      sidecarBytesMaterialized: true,
      webAssemblyValidated: false,
      abiConformanceVerified: false,
      sourceVerified: true,
      buildExecuted: false,
      outputIdentityAuthorized: false,
      reproducibilityVerified: false,
      releaseReady: false,
      factoryModuleDistributed: false,
    });
    expect(first.generatedWasmSha256).toBe(first.distributedWasmSha256);
    expect([...(await readFile(destination(input)))]).toEqual([...MINIMAL_WASM]);
    await expect(lstat(join(
      input.roots.outputRoot,
      "browsergrad-cpp-cute",
      "clang-extractor.mjs",
    ))).rejects.toMatchObject({ code: "ENOENT" });
    expect((await lstat(destination(input))).mode & 0o222).toBe(0);
  });

  it("rejects conflicting or symlink destinations without replacing them", async () => {
    const conflict = await fixture();
    const preparedConflict = await prepareCppCuteClangWasmBuildSource(conflict.input);
    await writeGeneratedSidecar(conflict.input);
    await mkdir(dirname(destination(conflict.input)), { recursive: true });
    await writeFile(destination(conflict.input), Uint8Array.of(1, 2, 3));
    await chmod(destination(conflict.input), 0o444);
    await expect(materializeCppCuteClangWasmSidecar(preparedConflict)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CONFLICT",
      path: "$.distributedWasmPath",
    });
    expect([...(await readFile(destination(conflict.input)))]).toEqual([1, 2, 3]);

    const linked = await fixture();
    const preparedLinked = await prepareCppCuteClangWasmBuildSource(linked.input);
    await writeGeneratedSidecar(linked.input);
    await mkdir(dirname(destination(linked.input)), { recursive: true });
    const linkTarget = join(linked.root, "link-target.wasm");
    await writeFile(linkTarget, MINIMAL_WASM);
    await symlink(linkTarget, destination(linked.input));
    await expect(materializeCppCuteClangWasmSidecar(preparedLinked)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CONFLICT",
      path: "$.distributedWasmPath",
    });
    expect([...(await readFile(linkTarget))]).toEqual([...MINIMAL_WASM]);
  });

  it("rejects writable identical destinations and unsafe output roots", async () => {
    const writable = await fixture();
    const preparedWritable = await prepareCppCuteClangWasmBuildSource(writable.input);
    await writeGeneratedSidecar(writable.input);
    await mkdir(dirname(destination(writable.input)), { recursive: true });
    await writeFile(destination(writable.input), MINIMAL_WASM, { mode: 0o644 });
    await expect(materializeCppCuteClangWasmSidecar(preparedWritable)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CONFLICT",
      path: "$.distributedWasmPath",
    });
    expect([...(await readFile(destination(writable.input)))]).toEqual([...MINIMAL_WASM]);

    const unsafe = await fixture();
    const preparedUnsafe = await prepareCppCuteClangWasmBuildSource(unsafe.input);
    await writeGeneratedSidecar(unsafe.input);
    await chmod(unsafe.input.roots.outputRoot, 0o777);
    await expect(materializeCppCuteClangWasmSidecar(preparedUnsafe)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: "$.outputRoot",
    });
    await expect(lstat(destination(unsafe.input))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("rejects a private output root beneath an untrusted writable ancestor", async () => {
    const { root, input } = await fixture();
    const unsafeAncestor = join(root, "unsafe-ancestor");
    const privateOutputRoot = join(unsafeAncestor, "private-output");
    await mkdir(privateOutputRoot, { recursive: true, mode: 0o700 });
    await chmod(unsafeAncestor, 0o777);
    (input.roots as { outputRoot: string }).outputRoot = privateOutputRoot;
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    await writeGeneratedSidecar(input);

    await expect(materializeCppCuteClangWasmSidecar(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: "$.outputRoot",
    });
    await expect(lstat(destination(input))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("rechecks the exact staged closure after source-byte verification", async () => {
    const { input } = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    await writeGeneratedSidecar(input);
    await chmod(input.roots.extractorSourceRoot, 0o700);
    await writeFile(join(input.roots.extractorSourceRoot, "undeclared.cpp"), "undeclared");
    await chmod(input.roots.extractorSourceRoot, 0o555);

    await expect(materializeCppCuteClangWasmSidecar(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: expect.stringContaining("$.stagedSourceRoot"),
    });
    await expect(lstat(destination(input))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("binds every staged source file to the inode captured during preparation", async () => {
    const { input } = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    await writeGeneratedSidecar(input);
    const stagedFile = join(input.roots.extractorSourceRoot, "CMakeLists.txt");
    const identicalBytes = await readFile(stagedFile);
    await chmod(input.roots.extractorSourceRoot, 0o700);
    await rm(stagedFile);
    await writeFile(stagedFile, identicalBytes, { mode: 0o444 });
    await chmod(input.roots.extractorSourceRoot, 0o555);

    await expect(materializeCppCuteClangWasmSidecar(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CONFLICT",
      path: expect.stringContaining("CMakeLists.txt"),
    });
    await expect(lstat(destination(input))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("rejects malformed generated bytes, forged source authority, and cancellation", async () => {
    const malformed = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(malformed.input);
    await writeGeneratedSidecar(malformed.input, Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8));
    await expect(materializeCppCuteClangWasmSidecar(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-INVALID",
      path: "$.generatedWasmPath",
    });
    const outputDirectory = dirname(destination(malformed.input));
    await expect(lstat(outputDirectory)).rejects.toMatchObject({ code: "ENOENT" });

    await expect(materializeCppCuteClangWasmSidecar({ ...prepared })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-UNVERIFIED",
    });

    await writeGeneratedSidecar(malformed.input);
    const controller = new AbortController();
    controller.abort();
    await expect(materializeCppCuteClangWasmSidecar(prepared, {
      signal: controller.signal,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CANCELLED",
    });

    const inFlight = await fixture();
    const inFlightPrepared = await prepareCppCuteClangWasmBuildSource(inFlight.input);
    await writeGeneratedSidecar(inFlight.input);
    const inFlightController = new AbortController();
    const pending = materializeCppCuteClangWasmSidecar(inFlightPrepared, {
      signal: inFlightController.signal,
    });
    inFlightController.abort();
    await expect(pending).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CANCELLED",
    });
    await expect(lstat(destination(inFlight.input))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("leaves no temporary output after a failed destination admission", async () => {
    const { input } = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    await writeGeneratedSidecar(input);
    const outputDirectory = dirname(destination(input));
    await mkdir(outputDirectory, { recursive: true });
    await writeFile(destination(input), Uint8Array.of(9));

    await expect(materializeCppCuteClangWasmSidecar(prepared)).rejects.toBeInstanceOf(
      CppCuteBrowserBuildExecutorError,
    );
    expect((await readdir(outputDirectory)).filter((entry) => entry.includes(".tmp-"))).toEqual([]);
  });

  it("rolls back the exact linked inode when post-link durability fails", async () => {
    const { input } = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    await writeGeneratedSidecar(input);
    const outputDirectory = dirname(destination(input));
    executorFsFaults.syncDirectoryFailure = {
      path: outputDirectory,
      remaining: 1,
      error: new Error("injected post-link directory sync failure"),
    };

    await expect(materializeCppCuteClangWasmSidecar(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-IO",
      path: "$.distributedWasmPath",
    });
    await expect(lstat(destination(input))).rejects.toMatchObject({ code: "ENOENT" });
    expect((await readdir(outputDirectory)).filter((entry) => entry.includes(".tmp-"))).toEqual([]);
  });

  it("refuses destructive rollback after a post-link destination identity swap", async () => {
    const { root, input } = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    await writeGeneratedSidecar(input);
    const displaced = join(root, "displaced-linked-sidecar.wasm");
    let replacementIdentity: { dev: bigint; ino: bigint } | undefined;
    executorFsFaults.afterLink = async (_sourcePath, destinationPath) => {
      await rename(destinationPath, displaced);
      await writeFile(destinationPath, MINIMAL_WASM, { mode: 0o444 });
      const stat = await lstat(destinationPath, { bigint: true });
      replacementIdentity = { dev: stat.dev, ino: stat.ino };
    };

    await expect(materializeCppCuteClangWasmSidecar(prepared)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CLEANUP",
      path: "$.distributedWasmPath",
    });
    const surviving = await lstat(destination(input), { bigint: true });
    expect({ dev: surviving.dev, ino: surviving.ino }).toEqual(replacementIdentity);
    expect([...(await readFile(destination(input)))]).toEqual([...MINIMAL_WASM]);
    expect((await readdir(dirname(destination(input))))
      .filter((entry) => entry.includes(".tmp-"))).toEqual([]);
  });

  it("settles temp removal and reports a typed composite close failure", async () => {
    const { input } = await fixture();
    const prepared = await prepareCppCuteClangWasmBuildSource(input);
    await writeGeneratedSidecar(input);
    const outputDirectory = dirname(destination(input));
    executorFsFaults.syncDirectoryFailure = {
      path: outputDirectory,
      remaining: 1,
      error: new Error("injected post-link directory sync failure"),
    };
    executorFsFaults.closeTemporaryFailure = new Error("injected temporary close failure");

    const observedError: unknown = await materializeCppCuteClangWasmSidecar(prepared)
      .then(() => undefined, (error: unknown) => error);
    expect(observedError).toBeInstanceOf(CppCuteBrowserBuildExecutorError);
    if (!(observedError instanceof CppCuteBrowserBuildExecutorError)) {
      throw new Error("expected typed executor cleanup failure");
    }
    expect(observedError).toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-EXECUTOR-CLEANUP",
      path: "$.distributedWasmPath",
    });
    expect(observedError.cause).toBeInstanceOf(AggregateError);
    if (!(observedError.cause instanceof AggregateError)) {
      throw new Error("expected aggregate operation and cleanup causes");
    }
    expect(observedError.cause.errors).toHaveLength(2);
    await expect(lstat(destination(input))).rejects.toMatchObject({ code: "ENOENT" });
    expect((await readdir(outputDirectory)).filter((entry) => entry.includes(".tmp-"))).toEqual([]);
  });
});
