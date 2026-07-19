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
      sha256: "652356abb90ad3ac80f2751e9db774185a3be529a93be597b3fe54921de910cf",
      byteLength: 577_876,
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
