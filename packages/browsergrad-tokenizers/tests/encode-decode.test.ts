import { describe, expect, it } from "vitest";
import {
  createByteBpeOracle,
  createByteBpeOracleModule,
  decodeByteBpe,
  deserializeByteBpeModel,
  encodeByteBpe,
  serializeByteBpeModel,
  trainByteBpe,
} from "../src/index";

describe("encodeByteBpe and decodeByteBpe", () => {
  it("roundtrips text through a trained byte BPE model", () => {
    const model = trainByteBpe(
      "low lower newest widest <|endoftext|> café café",
      {
        vocabSize: 280,
        specialTokens: ["<|endoftext|>"],
      },
    );

    const text = "low café <|endoftext|> widest";
    const ids = encodeByteBpe(text, model);

    expect(ids.length).toBeGreaterThan(0);
    expect(decodeByteBpe(ids, model)).toBe(text);
  });

  it("preserves special tokens as single IDs", () => {
    const model = trainByteBpe("hello <|endoftext|> world", {
      vocabSize: 270,
      specialTokens: ["<|endoftext|>"],
    });

    expect(encodeByteBpe("<|endoftext|>", model)).toEqual([256]);
  });

  it("serializes and deserializes models for bridge transport", () => {
    const model = trainByteBpe("the cat ate the cake", { vocabSize: 270 });
    const restored = deserializeByteBpeModel(serializeByteBpeModel(model));
    const text = "the cake";

    expect(encodeByteBpe(text, restored)).toEqual(encodeByteBpe(text, model));
    expect(decodeByteBpe(encodeByteBpe(text, restored), restored)).toBe(text);
  });

  it("byte-BPE oracle defaults to the GPT-2 end-of-text special token", () => {
    const oracle = createByteBpeOracle();
    const model = oracle.trainByteBpe("a<|endoftext|>b", { vocabSize: 258 });

    expect(model.specialTokens).toEqual(["<|endoftext|>"]);
    expect(model.merges).toEqual([]);
  });

  it("byte-BPE oracle module returns JSON-friendly serialized models", () => {
    const module = createByteBpeOracleModule();
    const model = module.train_byte_bpe("low lower <|endoftext|> widest", 270);
    const text = "lower <|endoftext|> widest";

    expect(model.specialTokens).toEqual(["<|endoftext|>"]);
    expect(module.decode_byte_bpe(module.encode_byte_bpe(text, model), model)).toBe(text);
  });
});
