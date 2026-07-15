import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  decodeLayoutArtifact,
  layoutArtifactPayload,
  traceViewCoordinate,
  verifyLayoutArtifact,
} from "../../src/layout";
import {
  canonicalizeJson,
  decodeWireJson,
  hashSemanticArtifact,
  isJsonObject,
  parseWireI64,
  sha256Hex,
  type JsonValue,
} from "../../src/schema";

const PACKAGE_ROOT = fileURLToPath(new URL("../../", import.meta.url));
const PYTHON_ORACLE = fileURLToPath(new URL("../../python/browsergrad_semantic_core.py", import.meta.url));
const FIXTURES = ["row-major-rank2", "symbolic-byte-rank3"] as const;

interface TraceCase {
  readonly viewOrdinal: number;
  readonly coordinates: readonly string[];
  readonly bindings: Readonly<Record<string, string>>;
}

describe("TypeScript/Python layout wire parity", () => {
  it.each(FIXTURES)("decodes, normalizes, canonicalizes, hashes, traces, and re-encodes %s identically", async (fixture) => {
    const inputFixture = fileURLToPath(new URL(`../../fixtures/layout-v1/${fixture}.input.json`, import.meta.url));
    const traceCases = fileURLToPath(new URL(`../../fixtures/layout-v1/${fixture}.cases.json`, import.meta.url));
    const expectedFixture = fileURLToPath(new URL(`../../fixtures/layout-v1/${fixture}.expected.json`, import.meta.url));
    const stdout = execFileSync("python3", [PYTHON_ORACLE, inputFixture, traceCases], {
      cwd: PACKAGE_ROOT,
      encoding: "utf8",
    });
    const pythonResult = decodeWireJson(new TextEncoder().encode(stdout));
    if (!isJsonObject(pythonResult)) throw new Error("Python parity result must be an object");

    const verified = await decodeLayoutArtifact(readFileSync(inputFixture));
    const payload = layoutArtifactPayload(verified);
    const rawEnvelope = decodeWireJson(readFileSync(inputFixture));
    if (!isJsonObject(rawEnvelope)) throw new Error("layout fixture must be an object");
    const normalizedEnvelope = { ...rawEnvelope, payload } as JsonValue;

    expect(canonicalizeJson(pythonResult.normalizedPayload)).toBe(canonicalizeJson(payload));
    expect(pythonResult.canonicalPayload).toBe(canonicalizeJson(payload));
    expect(pythonResult.canonicalArtifact).toBe(canonicalizeJson(normalizedEnvelope));
    expect(pythonResult.canonicalOrderingProbe)
      .toBe(canonicalizeJson({ "\ue000": 1, "\u{1f600}": 2, a: [2, 1] }));
    expect(pythonResult.semanticHash).toBe(await hashSemanticArtifact(verified));

    const casesValue = decodeWireJson(readFileSync(traceCases));
    if (!Array.isArray(casesValue)) throw new Error("trace fixture must be an array");
    const cases = casesValue as unknown as readonly TraceCase[];
    const traces = cases.map((entry) => {
      const viewId = payload.views[entry.viewOrdinal]?.viewId;
      if (viewId === undefined) throw new Error(`missing view ordinal ${entry.viewOrdinal}`);
      return traceViewCoordinate(verified, {
        viewId,
        coordinates: entry.coordinates.map((value) => parseWireI64(value)),
        bindings: Object.fromEntries(Object.entries(entry.bindings).map(([key, value]) => [key, parseWireI64(value)])),
      });
    });
    expect(canonicalizeJson(pythonResult.traces)).toBe(canonicalizeJson(traces));

    const expected = decodeWireJson(readFileSync(expectedFixture));
    if (!isJsonObject(expected) || typeof expected.semanticHash !== "string" || typeof expected.canonicalArtifactSha256 !== "string") {
      throw new Error("expected parity fixture must contain pinned hashes");
    }
    expect(pythonResult.semanticHash).toBe(expected.semanticHash);
    expect(await sha256Hex(new TextEncoder().encode(String(pythonResult.canonicalArtifact))))
      .toBe(expected.canonicalArtifactSha256);
    const traceSummaries = traces.map((trace) => ({
      mapLocation: trace.mapLocation.value,
      rootByteStart: trace.rootByteStart,
      rootByteEndExclusive: trace.rootByteEndExclusive,
      logicalInBounds: trace.logicalInBounds,
      predicateInBounds: trace.predicateInBounds,
      allocationInBounds: trace.allocationInBounds,
      accessInBounds: trace.accessInBounds,
    }));
    expect(canonicalizeJson(traceSummaries)).toBe(canonicalizeJson(expected.traces));

    const pythonEncodedArtifact = pythonResult.normalizedArtifact;
    const roundTrip = await verifyLayoutArtifact(pythonEncodedArtifact);
    expect(await hashSemanticArtifact(roundTrip)).toBe(await hashSemanticArtifact(verified));
  });

  it("rejects the same adversarial wire and semantic boundary corpus", async () => {
    const baseSource = readFileSync(fileURLToPath(new URL("../../fixtures/layout-v1/row-major-rank2.input.json", import.meta.url)), "utf8");
    const mutation = (): Record<string, unknown> => JSON.parse(baseSource) as Record<string, unknown>;
    const encode = (value: unknown): Uint8Array => new TextEncoder().encode(JSON.stringify(value));
    const cases: Array<{ readonly name: string; readonly bytes: Uint8Array }> = [];

    cases.push({
      name: "duplicate JSON key",
      bytes: new TextEncoder().encode(baseSource.replace(/^\s*\{/u, "{\"schema\":\"browsergrad.layout\",")),
    });
    cases.push({
      name: "invalid UTF-8",
      bytes: Uint8Array.from([0xff, 0xfe, 0xfd]),
    });
    cases.push({
      name: "unsafe JSON number",
      bytes: new TextEncoder().encode(baseSource.replace('"minor": 0', '"minor": 9007199254740992')),
    });

    const booleanVersion = mutation();
    (booleanVersion.version as Record<string, unknown>).major = true;
    cases.push({ name: "boolean version", bytes: encode(booleanVersion) });

    const unknownExtension = mutation();
    unknownExtension.requiredExtensions = ["test:unknown@1"];
    cases.push({ name: "unknown required extension", bytes: encode(unknownExtension) });

    const unknownField = mutation();
    (((unknownField.payload as Record<string, unknown>).views as Array<Record<string, unknown>>)[0] as Record<string, unknown>).newMeaning = true;
    cases.push({ name: "unknown closed field", bytes: encode(unknownField) });

    const misaligned = mutation();
    (((misaligned.payload as Record<string, unknown>).views as Array<Record<string, unknown>>)[0] as Record<string, unknown>).byteOffset = { kind: "const", value: "2" };
    cases.push({ name: "misaligned static offset", bytes: encode(misaligned) });

    const pastAllocation = mutation();
    (((pastAllocation.payload as Record<string, unknown>).views as Array<Record<string, unknown>>)[0] as Record<string, unknown>).byteOffset = { kind: "const", value: "28" };
    cases.push({ name: "offset past allocation", bytes: encode(pastAllocation) });

    const extentOverflow = mutation();
    (((extentOverflow.payload as Record<string, unknown>).views as Array<Record<string, unknown>>)[0] as Record<string, unknown>).shape = [{
      kind: "mul",
      lhs: { kind: "const", value: "4294967296" },
      rhs: { kind: "const", value: "4294967296" },
    }, { kind: "const", value: "3" }];
    cases.push({ name: "extent above u64", bytes: encode(extentOverflow) });

    const undeclared = mutation();
    (((undeclared.payload as Record<string, unknown>).views as Array<Record<string, unknown>>)[0] as Record<string, unknown>).shape = [
      { kind: "symbol", id: "missing" },
      { kind: "const", value: "3" },
    ];
    cases.push({ name: "undeclared symbol", bytes: encode(undeclared) });

    const divisor = mutation();
    const divisorPayload = divisor.payload as Record<string, unknown>;
    (divisorPayload.symbols as unknown[]) = [{ id: "d", domain: { min: "0", max: "4" } }];
    (((divisorPayload.views as Array<Record<string, unknown>>)[0] as Record<string, unknown>).shape as unknown[]) = [{
      kind: "floorDiv",
      value: { kind: "const", value: "4" },
      divisor: { kind: "symbol", id: "d" },
    }, { kind: "const", value: "3" }];
    cases.push({ name: "non-provably-positive divisor", bytes: encode(divisor) });

    const symbolSyntax = mutation();
    (symbolSyntax.payload as Record<string, unknown>).symbols = [{ id: "a:b", domain: { min: "1" } }];
    cases.push({ name: "invalid dimension symbol syntax", bytes: encode(symbolSyntax) });

    const reservedCoordinate = mutation();
    const reservedMap = (((reservedCoordinate.payload as Record<string, unknown>).indexMaps as Array<Record<string, unknown>>)[0] as Record<string, unknown>);
    reservedMap.location = { kind: "dimension", symbolId: "__bg_coordinate_0" };
    cases.push({ name: "reserved coordinate namespace", bytes: encode(reservedCoordinate) });

    const lowerBoundBomb = mutation();
    let bomb: Record<string, unknown> = { kind: "const", value: "9223372036854775807" };
    for (let depth = 0; depth < 8; depth += 1) {
      bomb = {
        kind: "mul",
        lhs: bomb,
        rhs: { kind: "const", value: "9223372036854775807" },
      };
    }
    (((lowerBoundBomb.payload as Record<string, unknown>).views as Array<Record<string, unknown>>)[0] as Record<string, unknown>).shape = [{
      kind: "floorDiv",
      value: { kind: "const", value: "4" },
      divisor: bomb,
    }, { kind: "const", value: "3" }];
    cases.push({ name: "divisor lower-bound integer bomb", bytes: encode(lowerBoundBomb) });

    const directory = mkdtempSync(join(tmpdir(), "browsergrad-layout-parity-"));
    try {
      const emptyCasesPath = join(directory, "empty.cases.json");
      writeFileSync(emptyCasesPath, "[]");
      for (const entry of cases) {
        let typescriptRejected = false;
        try {
          await decodeLayoutArtifact(entry.bytes);
        } catch {
          typescriptRejected = true;
        }
        expect(typescriptRejected, `${entry.name}: TypeScript accepted`).toBe(true);

        const inputPath = join(directory, `${cases.indexOf(entry)}.json`);
        writeFileSync(inputPath, entry.bytes);
        const python = spawnSync("python3", [PYTHON_ORACLE, inputPath, emptyCasesPath], {
          cwd: PACKAGE_ROOT,
          encoding: "utf8",
        });
        expect(python.status, `${entry.name}: Python accepted`).not.toBe(0);
      }
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("rejects the same dynamically resolved trace violations", async () => {
    const baseSource = readFileSync(fileURLToPath(new URL("../../fixtures/layout-v1/row-major-rank2.input.json", import.meta.url)), "utf8");
    const value = JSON.parse(baseSource) as Record<string, unknown>;
    const payload = value.payload as Record<string, unknown>;
    payload.symbols = [{ id: "offset", domain: { min: "0", max: "8" } }];
    (((payload.views as Array<Record<string, unknown>>)[0] as Record<string, unknown>).byteOffset) = { kind: "symbol", id: "offset" };
    const bytes = new TextEncoder().encode(JSON.stringify(value));
    const verified = await decodeLayoutArtifact(bytes);
    const viewId = layoutArtifactPayload(verified).views[0]?.viewId ?? "";
    expect(() => traceViewCoordinate(verified, {
      viewId,
      coordinates: [parseWireI64("0"), parseWireI64("0")],
      bindings: { offset: parseWireI64("2") },
    })).toThrowError(/BG-LAYOUT-INVALID-ALIGNMENT/u);

    const directory = mkdtempSync(join(tmpdir(), "browsergrad-layout-trace-parity-"));
    try {
      const inputPath = join(directory, "dynamic-offset.json");
      const casesPath = join(directory, "dynamic-offset.cases.json");
      writeFileSync(inputPath, bytes);
      writeFileSync(casesPath, JSON.stringify([{
        viewOrdinal: 0,
        coordinates: ["0", "0"],
        bindings: { offset: "2" },
      }]));
      const python = spawnSync("python3", [PYTHON_ORACLE, inputPath, casesPath], {
        cwd: PACKAGE_ROOT,
        encoding: "utf8",
      });
      expect(python.status, "Python accepted a dynamically misaligned view offset").not.toBe(0);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  it("resolves dominating predicates without irrelevant symbol bindings", async () => {
    const baseSource = readFileSync(fileURLToPath(new URL("../../fixtures/layout-v1/row-major-rank2.input.json", import.meta.url)), "utf8");
    const value = JSON.parse(baseSource) as Record<string, unknown>;
    const payload = value.payload as Record<string, unknown>;
    payload.symbols = [{ id: "n", domain: { min: "1", max: "4" } }];
    const indexMap = (payload.indexMaps as Array<Record<string, unknown>>)[0] as Record<string, unknown>;
    indexMap.inBounds = {
      kind: "and",
      values: [
        { kind: "bool", value: false },
        {
          kind: "lessEqual",
          lhs: { kind: "coordinate", axis: 0 },
          rhs: { kind: "dimension", symbolId: "n" },
        },
      ],
    };
    const bytes = new TextEncoder().encode(JSON.stringify(value));
    const verified = await decodeLayoutArtifact(bytes);
    const viewId = layoutArtifactPayload(verified).views[0]?.viewId ?? "";
    const typescriptTrace = traceViewCoordinate(verified, {
      viewId,
      coordinates: [parseWireI64("0"), parseWireI64("0")],
    });
    expect(typescriptTrace.predicateInBounds).toBe(false);

    const directory = mkdtempSync(join(tmpdir(), "browsergrad-layout-predicate-parity-"));
    try {
      const inputPath = join(directory, "dominating-predicate.json");
      const casesPath = join(directory, "dominating-predicate.cases.json");
      writeFileSync(inputPath, bytes);
      writeFileSync(casesPath, JSON.stringify([{
        viewOrdinal: 0,
        coordinates: ["0", "0"],
        bindings: {},
      }]));
      const python = spawnSync("python3", [PYTHON_ORACLE, inputPath, casesPath], {
        cwd: PACKAGE_ROOT,
        encoding: "utf8",
      });
      expect(python.status, python.stderr).toBe(0);
      const result = decodeWireJson(new TextEncoder().encode(python.stdout));
      if (!isJsonObject(result) || !Array.isArray(result.traces)) throw new Error("Python result missing traces");
      expect(canonicalizeJson(result.traces[0])).toBe(canonicalizeJson(typescriptTrace));
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
});
