import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { describe, expect, it } from "vitest";
import {
  buildCppCuteBrowserWorkerBundleProjection,
  renderCppCuteBrowserWorkerBundleResource,
} from "./cpp_cute_browser_worker_bundle_authoring.mjs";
import {
  CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE,
} from "../../dist/resources/cpp_cute_browser_reproducibility_v3.js";

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
      sha256: "d5a4519c806e63754f8e790c1fdb351cc4a030e5b9a9020f9d23751b3c4aeaac",
      byteLength: 584_660,
      staticImportCount: 0,
      dynamicImportCount: 0,
      factorySha256: "2eaa4ce31951cd5eff989679fd8d63c4ae74df0293f8f727209a3ce0f681764d",
      factoryByteLength: 27_884,
      verifierSha256: "81d5632dd0957b846897370e949d64beb40e6766bbae60999a041c56e20bf17f",
      verifierByteLength: 158_314,
    });
    expect(projection.factorySha256).toBe(
      CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE.builds[0].factoryModuleSha256,
    );
    expect(projection.factoryByteLength).toBe(
      CPP_CUTE_BROWSER_REPRODUCIBILITY_V3_RESOURCE.builds[0].factoryModuleByteLength,
    );
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
            sha256: "d5a4519c806e63754f8e790c1fdb351cc4a030e5b9a9020f9d23751b3c4aeaac",
            byteLength: 584_660,
          },
          {
            sha256: "81d5632dd0957b846897370e949d64beb40e6766bbae60999a041c56e20bf17f",
            byteLength: 158_314,
          },
        ],
      });
    }
  });
});
