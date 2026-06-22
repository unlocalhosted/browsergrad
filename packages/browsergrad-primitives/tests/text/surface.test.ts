import { describe, expect, it } from "vitest";
import {
  BYTE_BPE_EXAMPLE,
  GPT2_DEFAULT_SPECIAL_TOKENS,
  GPT2_PRETOKENIZER_PATTERN,
  byteBpeReferenceModule,
  createByteBpeReference,
  createByteBpeReferenceModule,
  createStreamingGate,
  decodeByteBpe,
  deserializeByteBpeModel,
  encodeByteBpe,
  serializeByteBpeModel,
  trainByteBpe,
} from "../../src/text";

describe("public surface", () => {
  it("exports byte-BPE defaults", () => {
    expect(GPT2_PRETOKENIZER_PATTERN).toContain("\\p{L}");
    expect(GPT2_DEFAULT_SPECIAL_TOKENS).toEqual(["<|endoftext|>"]);
    expect(BYTE_BPE_EXAMPLE.vocabSize).toBe(259);
  });

  it("exports planned helpers", () => {
    expect(typeof trainByteBpe).toBe("function");
    expect(typeof encodeByteBpe).toBe("function");
    expect(typeof decodeByteBpe).toBe("function");
    expect(typeof serializeByteBpeModel).toBe("function");
    expect(typeof deserializeByteBpeModel).toBe("function");
    expect(typeof createStreamingGate).toBe("function");
    expect(typeof createByteBpeReference().trainByteBpe).toBe("function");
    expect(typeof createByteBpeReferenceModule().train_byte_bpe).toBe("function");
    expect(typeof byteBpeReferenceModule.encode_byte_bpe).toBe("function");
  });
});
