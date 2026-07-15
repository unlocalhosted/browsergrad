import { describe, expect, it } from "vitest";

import {
  layoutArtifactPayload,
  prepareLayoutExpression,
  traceLayoutExpressionCoordinate,
  type DimExpr,
  type LayoutExpr,
  type PrepareLayoutExpressionRequest,
  type PreparedLayoutExpression,
} from "../../src/layout";
import {
  LAYOUT_DIAGNOSTIC_CODES,
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  parseWireI64,
  type WireI64,
} from "../../src/schema";

const wire = (value: string): WireI64 => parseWireI64(value);
const constant = (value: string): DimExpr => ({ kind: "const", value: wire(value) });
const symbol = (id: string): DimExpr => ({ kind: "symbol", id });

function strided(shape: readonly DimExpr[], strides: readonly DimExpr[]): LayoutExpr {
  return { kind: "strided", shape, strides };
}

function staticRequest(): PrepareLayoutExpressionRequest {
  return {
    symbols: [],
    constraints: [],
    layout: strided([constant("3"), constant("2")], [constant("1"), constant("3")]),
  };
}

function trace(prepared: PreparedLayoutExpression, coordinates: readonly string[]) {
  return traceLayoutExpressionCoordinate(prepared, {
    coordinates: coordinates.map(wire),
  });
}

async function asyncDiagnostic(run: () => Promise<unknown>): Promise<SemanticSchemaError> {
  try {
    await run();
    throw new Error("expected semantic failure");
  } catch (error) {
    expect(error).toBeInstanceOf(SemanticSchemaError);
    return error as SemanticSchemaError;
  }
}

function syncDiagnostic(run: () => unknown): SemanticSchemaError {
  try {
    run();
    throw new Error("expected semantic failure");
  } catch (error) {
    expect(error).toBeInstanceOf(SemanticSchemaError);
    return error as SemanticSchemaError;
  }
}

describe("standalone verified layout expressions", () => {
  it("traces the Gate 3 CuTe layout fixture without inventing storage", async () => {
    const prepared = await prepareLayoutExpression(staticRequest());
    const payload = layoutArtifactPayload(prepared.artifact);

    expect(Object.isFrozen(prepared)).toBe(true);
    expect(prepared.coordinateRank).toBe(2);
    expect(prepared.locationUnit).toBe("element");
    expect(prepared.layoutSemanticHash)
      .toBe("4e1fa226641c8441f503aa754b5c6d5bedc2449d9beb8987a7fa0cd222ce0667");
    expect(payload.allocations).toEqual([]);
    expect(payload.views).toEqual([]);
    expect(payload.indexMaps).toHaveLength(1);
    expect(payload.indexMaps[0]?.indexMapId).toBe(prepared.indexMapId);

    const coordinates = [
      ["0", "0"],
      ["0", "1"],
      ["1", "0"],
      ["1", "1"],
      ["2", "0"],
      ["2", "1"],
    ];
    expect(coordinates.map((coordinate) => trace(prepared, coordinate).mapLocation.value))
      .toEqual(["0", "3", "1", "4", "2", "5"]);
    expect(coordinates.map((coordinate) => trace(prepared, coordinate).layoutInBounds))
      .toEqual([true, true, true, true, true, true]);
  });

  it("reports logical and predicate bounds independently without byte-access claims", async () => {
    const prepared = await prepareLayoutExpression(staticRequest());
    expect(trace(prepared, ["-1", "0"])).toMatchObject({
      logicalCoordinates: ["-1", "0"],
      logicalShape: ["3", "2"],
      mapLocation: { unit: "element", value: "-1" },
      logicalInBounds: false,
      predicateInBounds: false,
      layoutInBounds: false,
    });
    expect(trace(prepared, ["3", "0"])).toMatchObject({
      mapLocation: { unit: "element", value: "3" },
      logicalInBounds: false,
      predicateInBounds: false,
      layoutInBounds: false,
    });
    expect(trace(prepared, ["0", "2"])).toMatchObject({
      mapLocation: { unit: "element", value: "6" },
      logicalInBounds: false,
      predicateInBounds: false,
      layoutInBounds: false,
    });
  });

  it("specializes dynamic shapes and constraints from explicit bindings", async () => {
    const prepared = await prepareLayoutExpression({
      symbols: [
        { id: "n", domain: { min: wire("1"), max: wire("8") } },
        { id: "m", domain: { min: wire("1"), max: wire("8") } },
      ],
      constraints: [
        { kind: "positive", value: symbol("m") },
        { kind: "positive", value: symbol("n") },
      ],
      layout: strided([symbol("m"), symbol("n")], [constant("1"), symbol("m")]),
    });
    const result = traceLayoutExpressionCoordinate(prepared, {
      coordinates: [wire("2"), wire("1")],
      bindings: { n: wire("2"), m: wire("3") },
    });
    expect(result).toMatchObject({
      logicalShape: ["3", "2"],
      mapLocation: { unit: "element", value: "5" },
      layoutInBounds: true,
    });

    expect(syncDiagnostic(() => traceLayoutExpressionCoordinate(prepared, {
      coordinates: [wire("2"), wire("1")],
      bindings: { m: wire("3") },
    })).diagnostic.code).toBe(LAYOUT_DIAGNOSTIC_CODES.unresolvedSymbol);
    expect(syncDiagnostic(() => traceLayoutExpressionCoordinate(prepared, {
      coordinates: [wire("2"), wire("1")],
      bindings: { n: wire("2"), m: wire("9") },
    })).diagnostic.code).toBe(LAYOUT_DIAGNOSTIC_CODES.symbolDomain);
  });

  it("sorts set-like symbols and constraints before deriving semantic identity", async () => {
    const m = { id: "m", domain: { min: wire("1"), max: wire("8") } };
    const n = { id: "n", domain: { min: wire("1"), max: wire("8") } };
    const positiveM = { kind: "positive" as const, value: symbol("m") };
    const positiveN = { kind: "positive" as const, value: symbol("n") };
    const layout = strided([symbol("m"), symbol("n")], [constant("1"), symbol("m")]);
    const first = await prepareLayoutExpression({ symbols: [m, n], constraints: [positiveM, positiveN], layout });
    const second = await prepareLayoutExpression({ symbols: [n, m], constraints: [positiveN, positiveM], layout });

    expect(first.layoutSemanticHash).toBe(second.layoutSemanticHash);
    expect(first.indexMapId).toBe(second.indexMapId);
  });

  it("keeps transport metadata outside semantic identity", async () => {
    const first = await prepareLayoutExpression(staticRequest(), {
      producer: { id: "frontend-a", version: "1" },
      artifactId: "transport-a",
    });
    const second = await prepareLayoutExpression(staticRequest(), {
      producer: { id: "frontend-b", version: "99" },
      artifactId: "transport-b",
    });
    expect(first.layoutSemanticHash).toBe(second.layoutSemanticHash);
    expect(first.indexMapId).toBe(second.indexMapId);
  });

  it("rejects structural copies and malformed trace requests", async () => {
    const prepared = await prepareLayoutExpression(staticRequest());
    expect(syncDiagnostic(() => traceLayoutExpressionCoordinate({ ...prepared } as PreparedLayoutExpression, {
      coordinates: [wire("0"), wire("0")],
    })).diagnostic.code).toBe(LAYOUT_DIAGNOSTIC_CODES.invalidArtifact);
    expect(syncDiagnostic(() => traceLayoutExpressionCoordinate(prepared, {
      coordinates: [wire("0")],
    })).diagnostic.code).toBe(LAYOUT_DIAGNOSTIC_CODES.invalidCoordinate);
    expect(syncDiagnostic(() => traceLayoutExpressionCoordinate(prepared, {
      coordinates: [wire("0"), wire("0")],
      extra: true,
    } as never)).diagnostic.code).toBe(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);
  });

  it("snapshots inputs and rejects hostile or open construction records", async () => {
    const request = staticRequest();
    const prepared = await prepareLayoutExpression(request);
    (request.layout as unknown as { shape: DimExpr[] }).shape[0] = constant("9");
    expect(trace(prepared, ["2", "1"]).mapLocation.value).toBe("5");

    expect((await asyncDiagnostic(() => prepareLayoutExpression({
      ...staticRequest(),
      extra: true,
    } as never))).diagnostic.code).toBe(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);
    expect((await asyncDiagnostic(() => prepareLayoutExpression(staticRequest(), {
      unknown: true,
    } as never))).diagnostic.code).toBe(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);

    const hostile = staticRequest() as PrepareLayoutExpressionRequest & { hidden?: boolean };
    Object.defineProperty(hostile, "hidden", { enumerable: true, get: () => true });
    expect((await asyncDiagnostic(() => prepareLayoutExpression(hostile))).diagnostic.code)
      .toBe(SCHEMA_DIAGNOSTIC_CODES.nonCanonicalValue);
  });

  it("applies caller resource limits before minting authority", async () => {
    expect((await asyncDiagnostic(() => prepareLayoutExpression(staticRequest(), {
      limits: { maxNodes: 4 },
    }))).diagnostic.code).toBe(SCHEMA_DIAGNOSTIC_CODES.resourceLimit);
  });
});
