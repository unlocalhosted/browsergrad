import { describe, expect, it } from "vitest";
import {
  BYTE_BPE_EXAMPLE,
  trainByteBpe,
  type BytePair,
} from "../../src/text";

const decoder = new TextDecoder();

function pairToText(pair: BytePair): readonly [string, string] {
  return [decoder.decode(pair[0]), decoder.decode(pair[1])];
}

describe("trainByteBpe", () => {
  it("uses the byte-BPE lexicographically greater tie-break", () => {
    const model = trainByteBpe(BYTE_BPE_EXAMPLE.corpus, {
      vocabSize: BYTE_BPE_EXAMPLE.vocabSize,
      specialTokens: BYTE_BPE_EXAMPLE.specialTokens,
    });

    expect(model.merges.map(pairToText).slice(0, 2)).toEqual([
      ["s", "t"],
      ["e", "st"],
    ]);
    expect(decoder.decode(model.vocab.get(257))).toBe("st");
    expect(decoder.decode(model.vocab.get(258))).toBe("est");
  });

  it("does not count pairs across special-token boundaries", () => {
    const boundaryOnlyCorpus = new Array(16)
      .fill(0)
      .map((_, i) => (i % 2 === 0 ? "a" : "b"))
      .join("<|endoftext|>");
    const model = trainByteBpe(boundaryOnlyCorpus, {
      vocabSize: 258,
      specialTokens: ["<|endoftext|>"],
    });

    expect(model.merges).toEqual([]);
    expect(model.vocab.size).toBe(257);
  });

  it("honors explicit tiny vocabulary limits", () => {
    const model = trainByteBpe("abc", { vocabSize: 2 });

    expect(model.vocab.size).toBe(2);
    expect(model.merges).toEqual([]);
    expect([...model.vocab.keys()]).toEqual([0, 1]);
  });

  it("rejects empty and duplicate special tokens", () => {
    expect(() =>
      trainByteBpe("abc", { vocabSize: 260, specialTokens: [""] }),
    ).toThrow(/special tokens must be non-empty/);

    expect(() =>
      trainByteBpe("abc", { vocabSize: 260, specialTokens: ["<x>", "<x>"] }),
    ).toThrow(/duplicate special token/);
  });
});
