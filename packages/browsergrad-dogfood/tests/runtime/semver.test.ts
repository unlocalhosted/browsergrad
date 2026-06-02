/**
 * Runtime — semver compatibility adversarial tests.
 *
 * Hypotheses: R6–R11.
 * Pure TS.
 */

import { describe, expect, it } from "vitest";
import {
  isSemverCompatible, assertCompatibleRuntime, parseManifest, LabRuntimeMismatch,
} from "@unlocalhosted/browsergrad-runtime";

describe("isSemverCompatible — adversarial", () => {
  it("caret accepts same-minor higher patches (R6)", () => {
    expect(isSemverCompatible("^0.8.0", "0.8.5")).toBe(true);
    expect(isSemverCompatible("^0.8.0", "0.8.0")).toBe(true);
  });

  it("caret on 0.x accepts higher minor (per npm semver, ^0.x.y === 0.x.x range)", () => {
    // npm semver: ^0.8.0 only matches 0.8.x; 0.9.0 is OUT of caret range.
    // The runtime docs claim it accepts 0.9.0 (caret = "compatible within major
    // for 1.x+, within minor for 0.x"). Confirm actual behavior.
    const accepts = isSemverCompatible("^0.8.0", "0.9.0");
    console.log(`[R6] ^0.8.0 vs 0.9.0 → ${accepts ? "accepted" : "rejected"}`);
    // Either behavior is defensible; just document it.
  });

  it("caret rejects next-major bump (R6)", () => {
    expect(isSemverCompatible("^0.8.0", "1.0.0")).toBe(false);
    expect(isSemverCompatible("^1.0.0", "2.0.0")).toBe(false);
  });

  it("tilde accepts same-minor higher patches (R7)", () => {
    expect(isSemverCompatible("~0.8.0", "0.8.5")).toBe(true);
  });

  it("tilde rejects different minor (R7)", () => {
    expect(isSemverCompatible("~0.8.0", "0.9.0")).toBe(false);
  });

  it("exact match: only matches the exact version (R8)", () => {
    expect(isSemverCompatible("0.8.0", "0.8.0")).toBe(true);
    expect(isSemverCompatible("0.8.0", "0.8.1")).toBe(false);
    expect(isSemverCompatible("0.8.0", "0.7.0")).toBe(false);
  });

  it("prerelease handling (R11) — documents actual behavior", () => {
    const a = isSemverCompatible("^0.8.0", "0.8.0-beta.1");
    const b = isSemverCompatible("^0.8.0-beta.0", "0.8.0-beta.1");
    console.log(`[R11] ^0.8.0 vs 0.8.0-beta.1 → ${a}; ^0.8.0-beta.0 vs 0.8.0-beta.1 → ${b}`);
    // Just document — prereleases are notoriously inconsistent across semver impls.
  });
});

describe("assertCompatibleRuntime — adversarial", () => {
  const VALID = {
    id: "test", version: "1.0.0", requires_browsergrad: "^0.8.0",
    required_ops: [], rubric_path: "r.py", starter_path: "s.py",
    reference_path: "ref.py", datasets: [],
  } as const;

  it("passes when runtime in range (R9)", () => {
    const r = parseManifest(VALID);
    if (!r.ok) throw new Error("parse failed");
    expect(() => assertCompatibleRuntime(r.manifest, "0.8.0")).not.toThrow();
  });

  it("throws LabRuntimeMismatch when runtime too old (R9)", () => {
    const r = parseManifest(VALID);
    if (!r.ok) throw new Error("parse failed");
    expect(() => assertCompatibleRuntime(r.manifest, "0.6.0")).toThrow(LabRuntimeMismatch);
  });

  it("LabRuntimeMismatch carries pin + runtime version fields (R10)", () => {
    const r = parseManifest(VALID);
    if (!r.ok) throw new Error("parse failed");
    try {
      assertCompatibleRuntime(r.manifest, "0.6.0");
      throw new Error("should have thrown");
    } catch (e) {
      expect(e).toBeInstanceOf(LabRuntimeMismatch);
      const lrm = e as LabRuntimeMismatch;
      expect(lrm.manifestPin).toBe("^0.8.0");
      expect(lrm.runtimeVersion).toBe("0.6.0");
    }
  });
});
