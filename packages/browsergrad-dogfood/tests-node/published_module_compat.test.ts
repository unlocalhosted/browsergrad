/**
 * Module-loader compatibility — does the published tarball import in raw Node?
 * The test below spawns a child Node process and tries to import each package
 * directly, catching published-tarball module metadata bugs that workspace tests
 * can miss.
 */

import { spawnSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { existsSync, writeFileSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function tryRawNodeImport(pkgName: string): { ok: boolean; stderr: string } {
  // Resolve the real package path via require.resolve-like mechanism: ask Node
  // itself, using the dogfood's own node_modules.
  const cwd = new URL("..", import.meta.url).pathname;
  const result = spawnSync(
    process.execPath,
    ["--input-type=module", "-e", `await import("${pkgName}"); console.log("ok");`],
    { cwd, encoding: "utf-8" },
  );
  return {
    ok: result.status === 0,
    stderr: (result.stderr || "").slice(0, 600),
  };
}

describe("raw Node ESM compat — published tarballs", () => {
  it("@unlocalhosted/browsergrad-kernels imports cleanly in raw Node ✓", () => {
    const r = tryRawNodeImport("@unlocalhosted/browsergrad-kernels");
    expect(r.ok, `stderr: ${r.stderr}`).toBe(true);
  });

  it("@unlocalhosted/browsergrad-runtime imports cleanly in raw Node ✓", () => {
    const r = tryRawNodeImport("@unlocalhosted/browsergrad-runtime");
    expect(r.ok, `stderr: ${r.stderr}`).toBe(true);
  });

  it("@unlocalhosted/browsergrad-compiler imports cleanly in raw Node ✓", () => {
    const r = tryRawNodeImport("@unlocalhosted/browsergrad-compiler");
    expect(r.ok, `stderr: ${r.stderr}`).toBe(true);
  });

  it("@unlocalhosted/browsergrad-grad imports cleanly in raw Node ✓ (fixed in 0.5.1)", () => {
    const r = tryRawNodeImport("@unlocalhosted/browsergrad-grad");
    expect(r.ok, `stderr: ${r.stderr}`).toBe(true);
  });

  it("@unlocalhosted/browsergrad-jit imports cleanly in raw Node ✓ (fixed in 0.8.1)", () => {
    const r = tryRawNodeImport("@unlocalhosted/browsergrad-jit");
    expect(r.ok, `stderr: ${r.stderr}`).toBe(true);
  });
});
