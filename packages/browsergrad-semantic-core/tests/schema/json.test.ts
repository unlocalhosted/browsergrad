import { describe, expect, it } from "vitest";

import {
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  assertJsonValue,
  decodeWireJson,
  parseWireI64,
  parseWireJson,
  parseWireU64,
  wireIntegerToBigInt,
} from "../../src/schema";

function expectCode(run: () => unknown, code: string): void {
  try {
    run();
    throw new Error("expected operation to fail");
  } catch (error) {
    expect(error).toBeInstanceOf(SemanticSchemaError);
    expect((error as SemanticSchemaError).diagnostic.code).toBe(code);
  }
}

describe("wire JSON", () => {
  it("parses a closed JSON tree without Object.prototype inheritance", () => {
    const parsed = parseWireJson('{"safe":1,"nested":[true,null,"x"]}');
    expect(parsed).toEqual({ safe: 1, nested: [true, null, "x"] });
    expect(Object.getPrototypeOf(parsed)).toBeNull();
  });

  it("rejects duplicate keys before semantic validation", () => {
    expectCode(
      () => parseWireJson('{"x":1,"x":2}'),
      SCHEMA_DIAGNOSTIC_CODES.duplicateKey,
    );
  });

  it.each(["1.0", "1e0", "-0", "9007199254740992"])(
    "rejects noncanonical or unsafe numeric lexeme %s",
    (lexeme) => {
      expectCode(() => parseWireJson(lexeme), SCHEMA_DIAGNOSTIC_CODES.unsafeNumber);
    },
  );

  it("rejects lone Unicode surrogates", () => {
    expectCode(() => parseWireJson('"\\ud800"'), SCHEMA_DIAGNOSTIC_CODES.invalidJson);
    expectCode(() => parseWireJson(`"${String.fromCharCode(0xd800)}"`), SCHEMA_DIAGNOSTIC_CODES.invalidJson);
  });

  it("enforces limits during parsing", () => {
    expectCode(
      () => parseWireJson("[1,2]", { limits: { maxArrayLength: 1 } }),
      SCHEMA_DIAGNOSTIC_CODES.resourceLimit,
    );
    expectCode(
      () => parseWireJson("[[]]", { limits: { maxDepth: 1 } }),
      SCHEMA_DIAGNOSTIC_CODES.resourceLimit,
    );
  });

  it("checks wire bytes before decoding and rejects invalid UTF-8", () => {
    expectCode(
      () => decodeWireJson(new Uint8Array([0xc3, 0x28])),
      SCHEMA_DIAGNOSTIC_CODES.invalidJson,
    );
    expectCode(
      () => decodeWireJson(new TextEncoder().encode("[1,2]"), { limits: { maxDocumentBytes: 4 } }),
      SCHEMA_DIAGNOSTIC_CODES.resourceLimit,
    );
  });

  it("rejects accessors and shared references in programmatic values", () => {
    const accessor = Object.defineProperty({}, "x", {
      enumerable: true,
      get: () => 1,
    });
    expectCode(() => assertJsonValue(accessor), SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);

    const shared = { x: 1 };
    expectCode(() => assertJsonValue([shared, shared]), SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);

    const arrayAccessor: unknown[] = [1];
    Object.defineProperty(arrayAccessor, "0", { enumerable: true, get: () => 1 });
    expectCode(() => assertJsonValue(arrayAccessor), SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);
  });

  it("applies the same node-count model to parsed and programmatic trees", () => {
    expect(() => parseWireJson('{"key":1}', { limits: { maxNodes: 2 } })).not.toThrow();
    expect(() => assertJsonValue({ key: 1 }, { limits: { maxNodes: 2 } })).not.toThrow();
  });
});

describe("wire integers", () => {
  it("round-trips signed and unsigned 64-bit boundaries", () => {
    expect(wireIntegerToBigInt(parseWireI64("-9223372036854775808"))).toBe(-(1n << 63n));
    expect(wireIntegerToBigInt(parseWireI64("9223372036854775807"))).toBe((1n << 63n) - 1n);
    expect(wireIntegerToBigInt(parseWireU64("18446744073709551615"))).toBe((1n << 64n) - 1n);
  });

  it.each(["+1", "-0", "01", " 1", "1e2", "١"])("rejects noncanonical integer %s", (value) => {
    expectCode(() => parseWireI64(value), SCHEMA_DIAGNOSTIC_CODES.nonCanonicalInteger);
  });

  it("rejects range overflow", () => {
    expectCode(() => parseWireI64("9223372036854775808"), SCHEMA_DIAGNOSTIC_CODES.integerRange);
    expectCode(() => parseWireU64("18446744073709551616"), SCHEMA_DIAGNOSTIC_CODES.integerRange);
  });
});
