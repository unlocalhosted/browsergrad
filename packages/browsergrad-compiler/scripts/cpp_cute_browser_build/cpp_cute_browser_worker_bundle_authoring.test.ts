import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import {
  buildCppCuteBrowserWorkerBundleProjection,
  renderCppCuteBrowserWorkerBundleResource,
} from "./cpp_cute_browser_worker_bundle_authoring.mjs";

const RESOURCE_URL = new URL(
  "../../src/resources/cpp_cute_browser_worker_bundle_v1.ts",
  import.meta.url,
);

describe("C++/CuTe package Worker bundle authoring", () => {
  it("emits one deterministic package-owned module graph with no imports", async () => {
    const projection = await buildCppCuteBrowserWorkerBundleProjection();
    expect(projection).toMatchObject({
      schema: "browsergrad.compiler.cpp-cute.package-worker-bundle-resource",
      version: 1,
      authority: "package-worker-bundle-authoring-projection-only",
      entryPath: "src/cpp_cute_browser_worker_module.ts",
      outputFileName: "browsergrad-cpp-cute-worker.mjs",
      sha256: "db0b0fd8f622d8c5febf7dc2c4daa75d68bf8be879f2d4f3d3670b56836f71d7",
      byteLength: 577_480,
      staticImportCount: 0,
      dynamicImportCount: 0,
      factorySha256: "796a548237420df7f5eca0c0260d3cbe752aeca155d9c7182c6ad0f5491dfb12",
      factoryByteLength: 27_125,
    });
    expect(projection.source).not.toContain("sourceMappingURL=");
    expect(renderCppCuteBrowserWorkerBundleResource(projection)).toBe(
      await readFile(RESOURCE_URL, "utf8"),
    );
  });
});
