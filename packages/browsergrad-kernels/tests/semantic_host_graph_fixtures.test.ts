import { describe, expect, it } from "vitest";

import {
  createVerifiedSignedReverseViewCopyArtifacts,
  patternedStorageBytes,
  reverseStorageElements,
  singleViewCopyGraphProgram,
} from "./semantic_host_graph_fixtures";

describe("semantic host-graph evidence fixtures", () => {
  it("constructs one canonical signed reverse layout without shared JSON nodes", async () => {
    const artifacts = await createVerifiedSignedReverseViewCopyArtifacts(
      "f64",
      8,
    );
    expect(artifacts.source.viewId).not.toBe(artifacts.destination.viewId);
    expect(artifacts.layoutSemanticHash).toMatch(/^[0-9a-f]{64}$/u);
    expect(artifacts.kernelSemanticHash).toMatch(/^[0-9a-f]{64}$/u);
  });

  it("rejects ranks outside the closed portable fixture profile", async () => {
    await expect(createVerifiedSignedReverseViewCopyArtifacts("i8", 0))
      .rejects.toThrow("rank must be an integer from 1 to 8");
    await expect(createVerifiedSignedReverseViewCopyArtifacts("i8", 9))
      .rejects.toThrow("rank must be an integer from 1 to 8");
  });

  it("bounds single-dispatch graph resources and rank multiplicity", async () => {
    const artifacts = await createVerifiedSignedReverseViewCopyArtifacts(
      "i8",
      8,
    );
    expect(() => singleViewCopyGraphProgram(artifacts, "i8", 3, 1))
      .toThrow("positive whole-word value");
    expect(() => singleViewCopyGraphProgram(artifacts, "i8", 256, 0))
      .toThrow("rank count must be from 1 to 64");
    expect(() => singleViewCopyGraphProgram(artifacts, "u8", 256, 1))
      .toThrow("must match both verified artifact roles");
    expect(() => singleViewCopyGraphProgram(artifacts, "i8", 4, 1))
      .toThrow("must match both verified artifact roles");
  });

  it("builds deterministic bounded byte patterns", () => {
    expect(patternedStorageBytes(4, 0x11))
      .toEqual(new Uint8Array([0x11, 0x36, 0x5b, 0x80]));
    expect(() => patternedStorageBytes(0, 0x11))
      .toThrow("storage fixture byte length");
    expect(() => patternedStorageBytes(4, 0x100))
      .toThrow("storage fixture seed must be one byte");
  });

  it("reverses complete elements without mutating input", () => {
    const input = new Uint8Array([1, 2, 3, 4, 5, 6]);
    expect(reverseStorageElements(input, 2))
      .toEqual(new Uint8Array([5, 6, 3, 4, 1, 2]));
    expect(input).toEqual(new Uint8Array([1, 2, 3, 4, 5, 6]));
    expect(() => reverseStorageElements(input, 4))
      .toThrow("complete bounded elements");
  });
});
