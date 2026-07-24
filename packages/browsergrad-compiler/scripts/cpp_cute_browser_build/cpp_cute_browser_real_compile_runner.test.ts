import {
  mkdtemp,
  mkdir,
  readFile,
  realpath,
  rm,
  stat,
  symlink,
  writeFile,
} from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, it } from "vitest";
import { canonicalJsonBytes } from
  "@unlocalhosted/browsergrad-semantic-core/schema";

import {
  CppCuteBrowserRealCompileRunnerError,
  persistCppCuteBrowserRealCompileEvidence,
  preflightCppCuteBrowserRealCompileInputs,
} from "./cpp_cute_browser_real_compile_runner.mjs";

const temporaryRoots: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map(
    (path) => rm(path, { recursive: true, force: true }),
  ));
});

describe("real browser C++/CuTe compile runner", () => {
  it("rejects unknown fields and relative paths before filesystem access", async () => {
    await expect(preflightCppCuteBrowserRealCompileInputs({
      wasmPath: "clang-extractor.wasm",
      packRoot: "/packs",
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-REAL-COMPILE-RUNNER",
      path: "$.wasmPath",
    });
    await expect(preflightCppCuteBrowserRealCompileInputs({
      wasmPath: "/clang-extractor.wasm",
      packRoot: "/packs",
      extra: true,
    } as never)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-REAL-COMPILE-RUNNER",
      path: "$.input",
    });
    await expect(preflightCppCuteBrowserRealCompileInputs({
      wasmPath: "/clang-extractor.wasm",
      packRoot: "/packs",
      caseId: "rank4",
    } as never)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-REAL-COMPILE-RUNNER",
      path: "$.caseId",
    });
  });

  it("rejects a symbolic Wasm path before reading external assets", async () => {
    const root = await temporaryRoot();
    const wasm = join(root, "clang-extractor.wasm");
    const alias = join(root, "clang-extractor-link.wasm");
    const packs = join(root, "packs");
    await Promise.all([
      writeFile(wasm, Uint8Array.of(0)),
      mkdir(packs),
    ]);
    await symlink(wasm, alias);

    await expect(preflightCppCuteBrowserRealCompileInputs({
      wasmPath: alias,
      packRoot: packs,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-REAL-COMPILE-RUNNER",
      path: "$.wasmPath",
    });
  });

  it("hashes the complete six-file set before rejecting an unpinned Wasm", async () => {
    const root = await temporaryRoot();
    const wasm = join(root, "clang-extractor.wasm");
    const packs = join(root, "packs");
    const assets = join(packs, "assets", "browsergrad-cpp-cute");
    await mkdir(assets, { recursive: true });
    await Promise.all([
      writeFile(wasm, Uint8Array.of(0, 97, 115, 109)),
      ...[
        "clang-resource.headers.bgvfs",
        "cuda-12.6.3.headers.bgvfs",
        "cutlass-3.7.0.headers.bgvfs",
        "libcxx-22.1.8.headers.bgvfs",
        "linux-sysroot.headers.bgvfs",
      ].map((name, index) => writeFile(join(assets, name), Uint8Array.of(index))),
    ]);

    await expect(preflightCppCuteBrowserRealCompileInputs({
      wasmPath: wasm,
      packRoot: packs,
    })).rejects.toEqual(expect.objectContaining({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-REAL-COMPILE-RUNNER",
      path: "$.assets[0]",
      message: expect.stringContaining("package-pinned two-clean-build output"),
    }) as Partial<CppCuteBrowserRealCompileRunnerError>);

    await expect(preflightCppCuteBrowserRealCompileInputs({
      wasmPath: wasm,
      packRoot: packs,
      allowUntrustedDiagnosticWasm: true,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-REAL-COMPILE-RUNNER",
      path: "$.assets[1]",
      message: expect.stringContaining(
        "header pack clang-resource differs from package reproducibility evidence",
      ),
    });

    await expect(preflightCppCuteBrowserRealCompileInputs({
      wasmPath: wasm,
      packRoot: packs,
      allowUntrustedDiagnosticWasm: true,
      requireCompiled: true,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-REAL-COMPILE-RUNNER",
      path: "$.assets[0]",
      message: expect.stringContaining("strict compile requires"),
    });
  });

  it("persists canonical immutable no-clobber evidence", async () => {
    const root = await temporaryRoot();
    const outputPath = join(root, "real-compile-evidence.json");
    const evidence = Object.freeze({
      schema: "browsergrad.compiler.cpp-cute.browser-real-compile-observation",
      version: 2,
      outcome: "compiled",
    });
    const written = await persistCppCuteBrowserRealCompileEvidence(
      outputPath,
      evidence,
    );
    const bytes = await readFile(outputPath);

    expect([...bytes]).toEqual([...canonicalJsonBytes(evidence)]);
    expect(written).toEqual({
      outputPath,
      sha256: expect.stringMatching(/^[0-9a-f]{64}$/u),
      byteLength: bytes.byteLength,
    });
    expect((await stat(outputPath)).mode & 0o222).toBe(0);
    await expect(persistCppCuteBrowserRealCompileEvidence(
      outputPath,
      evidence,
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-REAL-COMPILE-RUNNER",
      path: "$.evidenceOutput",
    });
  });
});

async function temporaryRoot(): Promise<string> {
  const root = await realpath(
    await mkdtemp(join(tmpdir(), "browsergrad-real-compile-runner.")),
  );
  temporaryRoots.push(root);
  return root;
}
