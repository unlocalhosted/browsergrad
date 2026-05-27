/**
 * Lab manifest + semver gate tests (PRD-013).
 */

import { describe, expect, it } from "vitest";
import {
  parseManifest,
  isSemverCompatible,
  assertCompatibleRuntime,
  LabRuntimeMismatch,
  type LabManifest,
} from "../src/lab";

const VALID_MANIFEST = {
  id: "single-neuron-relu",
  version: "1.0.0",
  requires_browsergrad: "^0.7.0",
  required_ops: ["MATMUL", "ADD", "WHERE"],
  rubric_path: "rubric.py",
  starter_path: "starter.py",
  reference_path: "reference.py",
  datasets: [],
};

describe("parseManifest", () => {
  it("accepts a minimal valid manifest", () => {
    const r = parseManifest(VALID_MANIFEST);
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.manifest.id).toBe("single-neuron-relu");
      expect(r.manifest.required_ops).toEqual(["MATMUL", "ADD", "WHERE"]);
    }
  });

  it("rejects non-object inputs", () => {
    expect(parseManifest("string").ok).toBe(false);
    expect(parseManifest(42).ok).toBe(false);
    expect(parseManifest(null).ok).toBe(false);
    expect(parseManifest([]).ok).toBe(false);
  });

  it("rejects ids that aren't kebab-case", () => {
    const r = parseManifest({ ...VALID_MANIFEST, id: "Bad_ID" });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.errors.some((e) => e.includes("id"))).toBe(true);
  });

  it("rejects malformed semver in version", () => {
    const r = parseManifest({ ...VALID_MANIFEST, version: "1.x" });
    expect(r.ok).toBe(false);
  });

  it("rejects malformed semver range in requires_browsergrad", () => {
    const r = parseManifest({ ...VALID_MANIFEST, requires_browsergrad: ">=0.7" });
    expect(r.ok).toBe(false);
  });

  it("rejects required_ops not being a string array", () => {
    const r = parseManifest({ ...VALID_MANIFEST, required_ops: "not-array" });
    expect(r.ok).toBe(false);
  });

  it("rejects too-long required_ops", () => {
    const r = parseManifest({
      ...VALID_MANIFEST,
      required_ops: new Array(65).fill("MATMUL"),
    });
    expect(r.ok).toBe(false);
  });

  it("accepts a manifest with datasets", () => {
    const r = parseManifest({
      ...VALID_MANIFEST,
      datasets: [
        { name: "mnist", url: "https://example.com/mnist.safetensors", hash: "sha256:abc" },
      ],
    });
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.manifest.datasets.length).toBe(1);
      expect(r.manifest.datasets[0]!.hash).toBe("sha256:abc");
    }
  });

  it("rejects malformed dataset entries", () => {
    const r = parseManifest({
      ...VALID_MANIFEST,
      datasets: [{ name: 123, url: "x" }],
    });
    expect(r.ok).toBe(false);
  });
});

describe("isSemverCompatible", () => {
  it("caret matches same major and version >= floor", () => {
    expect(isSemverCompatible("^0.7.0", "0.7.0")).toBe(true);
    expect(isSemverCompatible("^0.7.0", "0.7.5")).toBe(true);
    expect(isSemverCompatible("^0.7.0", "0.8.0")).toBe(true);
    expect(isSemverCompatible("^0.7.0", "0.6.9")).toBe(false);
    expect(isSemverCompatible("^0.7.0", "1.0.0")).toBe(false);
  });

  it("tilde matches same major+minor and version >= floor", () => {
    expect(isSemverCompatible("~0.7.0", "0.7.0")).toBe(true);
    expect(isSemverCompatible("~0.7.0", "0.7.5")).toBe(true);
    expect(isSemverCompatible("~0.7.0", "0.8.0")).toBe(false);
  });

  it("exact match", () => {
    expect(isSemverCompatible("0.7.0", "0.7.0")).toBe(true);
    expect(isSemverCompatible("0.7.0", "0.7.1")).toBe(false);
  });

  it("strips pre-release tags", () => {
    expect(isSemverCompatible("^0.7.0", "0.7.0-rc.1")).toBe(true);
  });

  it("fail-open on unparseable runtime", () => {
    expect(isSemverCompatible("^0.7.0", "garbage")).toBe(true);
  });
});

describe("assertCompatibleRuntime", () => {
  it("returns silently when compatible", () => {
    const m: LabManifest = {
      id: "x",
      version: "1.0.0",
      requires_browsergrad: "^0.7.0",
      required_ops: [],
      rubric_path: "r.py",
      starter_path: "s.py",
      reference_path: "ref.py",
      datasets: [],
    };
    expect(() => assertCompatibleRuntime(m, "0.7.3")).not.toThrow();
  });

  it("throws LabRuntimeMismatch when incompatible", () => {
    const m: LabManifest = {
      id: "x",
      version: "1.0.0",
      requires_browsergrad: "^0.7.0",
      required_ops: [],
      rubric_path: "r.py",
      starter_path: "s.py",
      reference_path: "ref.py",
      datasets: [],
    };
    expect(() => assertCompatibleRuntime(m, "0.6.0")).toThrow(LabRuntimeMismatch);
  });
});
