/**
 * Runtime — lab manifest validation adversarial tests.
 *
 * Hypotheses: R1–R5.
 * Pure TS (no Pyodide, no GPU) — runs in browser mode but also valid in node.
 */

import { describe, expect, it } from "vitest";
import { parseManifest } from "@unlocalhosted/browsergrad-runtime";

const VALID = {
  id: "single-neuron-relu",
  version: "1.0.0",
  requires_browsergrad: "^0.8.0",
  required_ops: ["MATMUL", "ADD", "WHERE"],
  rubric_path: "rubric.py",
  starter_path: "starter.py",
  reference_path: "reference.py",
  datasets: [],
} as const;

describe("parseManifest — adversarial", () => {
  it("accepts a minimal valid manifest (R1)", () => {
    const r = parseManifest(VALID);
    expect(r.ok).toBe(true);
  });

  it("rejects non-kebab-case id (R2)", () => {
    expect(parseManifest({ ...VALID, id: "Bad_ID" }).ok).toBe(false);
    expect(parseManifest({ ...VALID, id: "has spaces" }).ok).toBe(false);
    expect(parseManifest({ ...VALID, id: "" }).ok).toBe(false);
  });

  it("rejects malformed semver in version (R3)", () => {
    expect(parseManifest({ ...VALID, version: "1.x" }).ok).toBe(false);
    expect(parseManifest({ ...VALID, version: "not-a-version" }).ok).toBe(false);
    expect(parseManifest({ ...VALID, version: "" }).ok).toBe(false);
  });

  it("rejects oversized required_ops (>64 entries) (R4)", () => {
    const tooLong = { ...VALID, required_ops: new Array(65).fill("X") };
    expect(parseManifest(tooLong).ok).toBe(false);
  });

  it("rejects wrong-type required_ops (R5)", () => {
    const wrongType = { ...VALID, required_ops: "not-an-array" };
    expect(parseManifest(wrongType).ok).toBe(false);
  });

  it("rejects missing required fields", () => {
    const { id: _id, ...missingId } = VALID;
    expect(parseManifest(missingId).ok).toBe(false);
    const { version: _v, ...missingVersion } = VALID;
    expect(parseManifest(missingVersion).ok).toBe(false);
    const { rubric_path: _r, ...missingRubric } = VALID;
    expect(parseManifest(missingRubric).ok).toBe(false);
  });

  it("rejects entirely-malformed input", () => {
    expect(parseManifest(null).ok).toBe(false);
    expect(parseManifest(undefined).ok).toBe(false);
    expect(parseManifest("not-an-object").ok).toBe(false);
    expect(parseManifest(42).ok).toBe(false);
    expect(parseManifest([]).ok).toBe(false);
  });
});
