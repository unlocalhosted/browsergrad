import { describe, expect, it } from "vitest";

import {
  DENSE_PERMUTATION_VIEW_COPY_FIXTURES,
  decodeDensePermutationViewCopyFixtures,
  fixtureWords,
} from "../../../../test-support/dense-permutation-view-copy-fixtures";
import {
  createVerifiedDensePermutationViewCopyArtifacts,
  prepareViewCopyCpu,
} from "../../src/kernel";
import { parseWireI64 } from "../../src/schema";

const EXPECTED_CASE_IDS = ["rank2-transpose", "rank3-permutation"] as const;

describe("dense-permutation shared fixtures", () => {
  it("keeps a closed, deeply immutable, complete fixture set", () => {
    expect(DENSE_PERMUTATION_VIEW_COPY_FIXTURES.cases.map(({ id }) => id))
      .toEqual(EXPECTED_CASE_IDS);
    expect(Object.isFrozen(DENSE_PERMUTATION_VIEW_COPY_FIXTURES)).toBe(true);
    expect(Object.isFrozen(DENSE_PERMUTATION_VIEW_COPY_FIXTURES.version)).toBe(true);
    expect(Object.isFrozen(DENSE_PERMUTATION_VIEW_COPY_FIXTURES.cases)).toBe(true);
    for (const fixture of DENSE_PERMUTATION_VIEW_COPY_FIXTURES.cases) {
      expect(Object.isFrozen(fixture)).toBe(true);
      expect(Object.isFrozen(fixture.request)).toBe(true);
      expect(Object.isFrozen(fixture.request.inputShape)).toBe(true);
      expect(Object.isFrozen(fixture.request.axes)).toBe(true);
      expect(Object.isFrozen(fixture.sourceWords)).toBe(true);
      expect(Object.isFrozen(fixture.expectedOutputWords)).toBe(true);
    }
  });

  it.each(DENSE_PERMUTATION_VIEW_COPY_FIXTURES.cases)(
    "reconstructs and executes $id with pinned hashes and exact u32 bits",
    async (fixture) => {
      const artifacts = await createVerifiedDensePermutationViewCopyArtifacts({
        inputShape: fixture.request.inputShape.map((extent) => parseWireI64(extent)),
        axes: fixture.request.axes,
        dtype: fixture.request.dtype,
      }, { producer: { id: "shared-fixture-proof", version: "1" } });
      const prepared = await prepareViewCopyCpu(artifacts.layout, artifacts.kernel, {
        operationId: artifacts.operationId,
      });
      const source = fixtureWords(fixture.sourceWords);
      const destination = new Uint32Array(fixture.expectedOutputWords.length);

      prepared.execute({
        source: bytes(source),
        destination: bytes(destination),
      });

      expect(artifacts.layoutSemanticHash).toBe(fixture.layoutSemanticHash);
      expect(artifacts.kernelSemanticHash).toBe(fixture.kernelSemanticHash);
      expect([...destination]).toEqual([...fixtureWords(fixture.expectedOutputWords)]);
    },
  );

  it("rejects open or internally inconsistent fixture mutations", () => {
    const base = structuredClone(DENSE_PERMUTATION_VIEW_COPY_FIXTURES);
    expect(() => decodeDensePermutationViewCopyFixtures({ ...base, extra: true }))
      .toThrow(/expected closed fields/);

    const wrongOutput = {
      ...base,
      cases: base.cases.map((fixture, index) => index === 0
        ? { ...fixture, expectedOutputWords: ["00000000", ...fixture.expectedOutputWords.slice(1)] }
        : fixture),
    };
    expect(() => decodeDensePermutationViewCopyFixtures(wrongOutput))
      .toThrow(/must copy source word/);

    const noncanonicalWord = {
      ...base,
      cases: base.cases.map((fixture, index) => index === 0
        ? { ...fixture, sourceWords: ["3F800000", ...fixture.sourceWords.slice(1)] }
        : fixture),
    };
    expect(() => decodeDensePermutationViewCopyFixtures(noncanonicalWord))
      .toThrow(/eight lowercase hexadecimal digits/);
  });
});

function bytes(words: Uint32Array): Uint8Array {
  return new Uint8Array(words.buffer, words.byteOffset, words.byteLength);
}
