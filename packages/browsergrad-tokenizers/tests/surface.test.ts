import { describe, expect, it } from "vitest";
import {
  CS336_BPE_EXAMPLE,
  CS336_DEFAULT_SPECIAL_TOKENS,
  CS336_PRETOKENIZER_PATTERN,
  createCs336TokenizerOracle,
  createStreamingGate,
  decodeByteBpe,
  deserializeByteBpeModel,
  encodeByteBpe,
  serializeByteBpeModel,
  trainByteBpe,
} from "../src/index";

describe("public surface", () => {
  it("exports CS336 defaults", () => {
    expect(CS336_PRETOKENIZER_PATTERN).toContain("\\p{L}");
    expect(CS336_DEFAULT_SPECIAL_TOKENS).toEqual(["<|endoftext|>"]);
    expect(CS336_BPE_EXAMPLE.vocabSize).toBe(259);
  });

  it("exports planned helpers", () => {
    expect(typeof trainByteBpe).toBe("function");
    expect(typeof encodeByteBpe).toBe("function");
    expect(typeof decodeByteBpe).toBe("function");
    expect(typeof serializeByteBpeModel).toBe("function");
    expect(typeof deserializeByteBpeModel).toBe("function");
    expect(typeof createStreamingGate).toBe("function");
    expect(typeof createCs336TokenizerOracle().trainByteBpe).toBe("function");
  });
});
