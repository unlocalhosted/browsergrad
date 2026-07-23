import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { describe, expect, it } from "vitest";
import {
  buildCppCuteBrowserWorkerBundleProjection,
  renderCppCuteBrowserWorkerBundleResource,
} from "./cpp_cute_browser_worker_bundle_authoring.mjs";

const RESOURCE_URL = new URL(
  "../../src/resources/cpp_cute_browser_worker_bundle_v1.ts",
  import.meta.url,
);
const AUTHORING_SCRIPT_PATH = fileURLToPath(new URL(
  "./cpp_cute_browser_worker_bundle_authoring.mjs",
  import.meta.url,
));
const PACKAGE_DIRECTORY = fileURLToPath(new URL("../../", import.meta.url));
const REPOSITORY_DIRECTORY = fileURLToPath(new URL("../../../../", import.meta.url));
const execFileAsync = promisify(execFile);

describe("C++/CuTe package Worker bundle authoring", () => {
  it("emits one deterministic package-owned module graph with no imports", async () => {
    const projection = await buildCppCuteBrowserWorkerBundleProjection();
    expect(projection).toMatchObject({
      schema: "browsergrad.compiler.cpp-cute.package-worker-bundle-resource",
      version: 1,
      authority: "package-worker-bundle-authoring-projection-only",
      entryPath: "src/cpp_cute_browser_worker_module.ts",
      outputFileName: "browsergrad-cpp-cute-worker.mjs",
      sha256: "c92801cb50a320a0b82ca970dc6489b9eff28cca8e1e0e4bfdf1dd7265208f74",
      byteLength: 581_489,
      staticImportCount: 0,
      dynamicImportCount: 0,
      factorySha256: "f64d5239d5c258f44e859834b57e1ea330b7efdf7a405dead3126b53330a5534",
      factoryByteLength: 27_285,
      verifierSha256: "899b1890142dcc68fcceb4b4e08fa215f36d805240fabe3f655f646f39364fa2",
      verifierByteLength: 156_402,
    });
    expect(projection.source).not.toContain("sourceMappingURL=");
    expect(projection.source).not.toContain("//#region");
    expect(renderCppCuteBrowserWorkerBundleResource(projection)).toBe(
      await readFile(RESOURCE_URL, "utf8"),
    );
  });

  it("checks the same canonical bytes from the package and repository directories", async () => {
    for (const cwd of [PACKAGE_DIRECTORY, REPOSITORY_DIRECTORY]) {
      const { stdout } = await execFileAsync(
        process.execPath,
        [AUTHORING_SCRIPT_PATH, "--check"],
        { cwd },
      );
      expect(JSON.parse(stdout)).toMatchObject({
        authority: "package-worker-bundle-set-authoring-projection-only",
        checked: true,
        bundles: [
          {
            sha256: "c92801cb50a320a0b82ca970dc6489b9eff28cca8e1e0e4bfdf1dd7265208f74",
            byteLength: 581_489,
          },
          {
            sha256: "899b1890142dcc68fcceb4b4e08fa215f36d805240fabe3f655f646f39364fa2",
            byteLength: 156_402,
          },
        ],
      });
    }
  });
});
