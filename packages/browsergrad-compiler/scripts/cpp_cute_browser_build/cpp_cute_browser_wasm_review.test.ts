import { createHash } from "node:crypto";
import {
  chmod,
  lstat,
  mkdtemp,
  readFile,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { canonicalJsonBytes } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { afterEach, describe, expect, it } from "vitest";

import {
  parseCppCuteBrowserWasmReviewArguments,
  requireExactCppCuteBrowserWasmInterface,
  reviewCppCuteBrowserWasmFile,
  writeCppCuteBrowserWasmReviewReport,
} from "./cpp_cute_browser_wasm_review.mjs";

const MINIMAL_WASM = Uint8Array.of(0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00);
const temporaryRoots: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "browsergrad-wasm-review-"));
  temporaryRoots.push(root);
  const wasmPath = join(root, "clang-extractor.wasm");
  await writeFile(wasmPath, MINIMAL_WASM);
  return { root, wasmPath };
}

describe("build-produced Clang-Wasm review report", () => {
  it("emits bounded raw-module discovery without authorizing a mismatch", async () => {
    const { wasmPath } = await fixture();
    const report = await reviewCppCuteBrowserWasmFile({ wasmPath });

    expect(report).toMatchObject({
      authority: "review-observation-only",
      wasmSha256: createHash("sha256").update(MINIMAL_WASM).digest("hex"),
      wasmByteLength: MINIMAL_WASM.byteLength,
      exactInterfaceConformance: false,
      rawWasmVerified: true,
      workerExecutionReady: false,
      releaseReady: false,
    });
    expect(report.mismatches.length).toBeGreaterThan(0);
    expect(report.projection).toMatchObject({
      sectionOrder: [],
      imports: [],
      exports: [],
      memories: [],
    });
  });

  it("rejects malformed modules and symlink inputs", async () => {
    const malformed = await fixture();
    await writeFile(malformed.wasmPath, Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8));
    await expect(reviewCppCuteBrowserWasmFile({ wasmPath: malformed.wasmPath })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-INVALID",
    });

    const linked = await fixture();
    const linkPath = join(linked.root, "linked.wasm");
    await symlink(linked.wasmPath, linkPath);
    await expect(reviewCppCuteBrowserWasmFile({ wasmPath: linkPath })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-REVIEW-INVALID",
      path: "$.wasmPath",
    });
  });

  it("writes canonical no-clobber read-only reports", async () => {
    const { root, wasmPath } = await fixture();
    const report = await reviewCppCuteBrowserWasmFile({ wasmPath });
    const outputPath = join(root, "wasm-review.v1.json");
    const result = await writeCppCuteBrowserWasmReviewReport(outputPath, report);
    const bytes = await readFile(outputPath);

    expect([...bytes]).toEqual([...canonicalJsonBytes(report)]);
    expect(result).toEqual({
      outputPath,
      sha256: createHash("sha256").update(bytes).digest("hex"),
      byteLength: bytes.byteLength,
      exactInterfaceConformance: false,
      mismatchCount: report.mismatches.length,
    });
    expect((await lstat(outputPath)).mode & 0o222).toBe(0);
    await expect(writeCppCuteBrowserWasmReviewReport(outputPath, report)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-REVIEW-CONFLICT",
      path: "$outputPath",
    });
    await chmod(outputPath, 0o600);
  });

  it("refuses to serialize forged review objects", async () => {
    const { root, wasmPath } = await fixture();
    const report = await reviewCppCuteBrowserWasmFile({ wasmPath });
    await expect(writeCppCuteBrowserWasmReviewReport(
      join(root, "forged.json"),
      { ...report },
    )).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-REVIEW-INVALID",
      path: "$report",
    });
  });

  it("turns an authentic mismatch observation into a strict validation failure", async () => {
    const { wasmPath } = await fixture();
    const report = await reviewCppCuteBrowserWasmFile({ wasmPath });

    expect(() => requireExactCppCuteBrowserWasmInterface(report)).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-REVIEW-MISMATCH",
        path: "$report.exactInterfaceConformance",
      }),
    );
    expect(() => requireExactCppCuteBrowserWasmInterface({ ...report })).toThrowError(
      expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-WASM-REVIEW-INVALID",
        path: "$report",
      }),
    );
  });
});

describe("Clang-Wasm review CLI arguments", () => {
  it("admits only the exact two absolute named paths", () => {
    expect(parseCppCuteBrowserWasmReviewArguments([
      "--wasm=/work/clang-extractor.wasm",
      "--output=/work/review.json",
    ])).toEqual({
      output: "/work/review.json",
      requireExactInterface: false,
      wasm: "/work/clang-extractor.wasm",
    });
    expect(parseCppCuteBrowserWasmReviewArguments([
      "--require-exact-interface",
      "--wasm=/work/clang-extractor.wasm",
      "--output=/work/review.json",
    ])).toEqual({
      output: "/work/review.json",
      requireExactInterface: true,
      wasm: "/work/clang-extractor.wasm",
    });
    expect(() => parseCppCuteBrowserWasmReviewArguments([
      "--wasm=/work/clang-extractor.wasm",
    ])).toThrow();
    expect(() => parseCppCuteBrowserWasmReviewArguments([
      "--wasm=relative.wasm",
      "--output=/work/review.json",
    ])).toThrow();
  });
});
