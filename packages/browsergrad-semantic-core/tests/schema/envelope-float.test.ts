import { describe, expect, it } from "vitest";

import {
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  parseFloatBits,
  parseWireJson,
  validateWireEnvelope,
} from "../../src/schema";

function diagnosticCode(run: () => unknown): string {
  try {
    run();
    throw new Error("expected operation to fail");
  } catch (error) {
    expect(error).toBeInstanceOf(SemanticSchemaError);
    return (error as SemanticSchemaError).diagnostic.code;
  }
}

describe("wire envelope", () => {
  it("normalizes extension order and deep-freezes the result", () => {
    const value = parseWireJson('{"schema":"browsergrad.layout","version":{"major":1,"minor":2},"producer":{"id":"tests","version":"1"},"artifactId":"a","payload":{"x":1},"requiredExtensions":["org.example:z@1","org.example:a@1"]}');
    const result = validateWireEnvelope(value, {
      schema: "browsergrad.layout",
      supportedMajor: 1,
      supportedMinor: 0,
      knownRequiredExtensions: new Set(["org.example:a@1", "org.example:z@1"]),
    });

    expect(result.requiredExtensions).toEqual(["org.example:a@1", "org.example:z@1"]);
    expect(Object.isFrozen(result)).toBe(true);
    expect(Object.isFrozen(result.payload)).toBe(true);
  });

  it("rejects unknown major, required extension, and closed-record fields", () => {
    const major = parseWireJson('{"schema":"browsergrad.layout","version":{"major":2,"minor":0},"producer":{"id":"tests","version":"1"},"artifactId":"a","payload":{},"requiredExtensions":[]}');
    expect(diagnosticCode(() => validateWireEnvelope(major, { schema: "browsergrad.layout", supportedMajor: 1, supportedMinor: 0 })))
      .toBe(SCHEMA_DIAGNOSTIC_CODES.unsupportedVersion);

    const extension = parseWireJson('{"schema":"browsergrad.layout","version":{"major":1,"minor":0},"producer":{"id":"tests","version":"1"},"artifactId":"a","payload":{},"requiredExtensions":["org.example:new@1"]}');
    expect(diagnosticCode(() => validateWireEnvelope(extension, { schema: "browsergrad.layout", supportedMajor: 1, supportedMinor: 0 })))
      .toBe(SCHEMA_DIAGNOSTIC_CODES.unknownRequiredExtension);

    const unknown = parseWireJson('{"schema":"browsergrad.layout","version":{"major":1,"minor":0},"producer":{"id":"tests","version":"1"},"artifactId":"a","payload":{},"requiredExtensions":[],"newSemanticField":1}');
    expect(diagnosticCode(() => validateWireEnvelope(unknown, { schema: "browsergrad.layout", supportedMajor: 1, supportedMinor: 0 })))
      .toBe(SCHEMA_DIAGNOSTIC_CODES.invalidEnvelope);
  });

  it("rejects non-JSON programmatic envelopes before semantic validation", () => {
    const value = Object.create({ inherited: true }) as Record<string, unknown>;
    value.schema = "browsergrad.layout";
    expect(diagnosticCode(() => validateWireEnvelope(value, {
      schema: "browsergrad.layout",
      supportedMajor: 1,
      supportedMinor: 0,
    }))).toBe(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);
  });
});

describe("float bit records", () => {
  it.each([
    ["f16", "8000"],
    ["bf16", "7fc1"],
    ["f32", "ff800000"],
    ["f64", "7ff8000000000001"],
  ] as const)("preserves %s bit identity", (dtype, bits) => {
    expect(parseFloatBits({ kind: "float-bits", dtype, bits })).toEqual({ kind: "float-bits", dtype, bits });
  });

  it("rejects noncanonical widths and uppercase hex", () => {
    expect(diagnosticCode(() => parseFloatBits({ kind: "float-bits", dtype: "f32", bits: "0" })))
      .toBe(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);
    expect(diagnosticCode(() => parseFloatBits({ kind: "float-bits", dtype: "f16", bits: "7FC0" })))
      .toBe(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);
  });

  it("does not execute accessors while validating float records", () => {
    const value = Object.defineProperty({}, "kind", {
      enumerable: true,
      get: () => "float-bits",
    });
    expect(diagnosticCode(() => parseFloatBits(value))).toBe(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);
  });
});
