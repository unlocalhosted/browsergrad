import { describe, expect, it } from "vitest";

import {
  parseCppCuteBrowserBuildLockAuthoringArguments,
  projectCppCuteBrowserBuildInputLock,
} from "./cpp_cute_browser_build_lock_authoring.mjs";

describe("Clang-Wasm build-lock authoring projection", () => {
  it("derives the complete checked-in source and lock identity without editing it", async () => {
    const report = await projectCppCuteBrowserBuildInputLock();

    expect(report).toMatchObject({
      schema: "browsergrad.compiler.cpp-cute.browser-build-lock-authoring-projection",
      version: 1,
      authority: "authoring-projection-only",
      checkedInResourceMatches: true,
    });
    expect(report.files).toHaveLength(42);
    const paths = report.files.map((file) => file.path);
    expect(paths).toEqual(paths.slice().sort());
    expect(report.lockId).toMatch(/^bg\.cpp\.browser-build-input-lock\.sha256\.[0-9a-f]{64}$/u);
    expect(report.resourceSha256).toMatch(/^[0-9a-f]{64}$/u);
    expect(report.recipeSha256).toMatch(/^[0-9a-f]{64}$/u);
    expect(report.extractorSourceSetSha256).toMatch(/^[0-9a-f]{64}$/u);
  });

  it("admits only print and check modes", () => {
    expect(parseCppCuteBrowserBuildLockAuthoringArguments([])).toEqual({ check: false });
    expect(parseCppCuteBrowserBuildLockAuthoringArguments(["--check"])).toEqual({ check: true });
    expect(() => parseCppCuteBrowserBuildLockAuthoringArguments(["--write"]))
      .toThrowError(expect.objectContaining({
        code: "BG-COMPILER-CPP-CUTE-BROWSER-BUILD-LOCK-AUTHORING",
        path: "$arguments",
      }));
  });
});
