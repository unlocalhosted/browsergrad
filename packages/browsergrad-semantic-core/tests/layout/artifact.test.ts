import { describe, expect, it } from "vitest";

import {
  decodeLayoutArtifact,
  layoutArtifactPayload,
  traceViewAlias,
  traceViewCoordinate,
  verifyLayoutArtifact,
  type VerifiedLayoutArtifact,
} from "../../src/layout";
import { unwrapLayoutArtifact } from "../../src/layout/artifact";
import {
  LAYOUT_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  hashSemanticArtifact,
  parseWireI64,
} from "../../src/schema";

function artifact(local = { allocation: "alloc", alias: "alias", indexMap: "map", view: "view" }): Record<string, unknown> {
  return {
    schema: "browsergrad.layout",
    version: { major: 1, minor: 0 },
    producer: { id: "tests", version: "1" },
    artifactId: "transport-id",
    requiredExtensions: [],
    payload: {
      symbols: [],
      constraints: [],
      allocations: [{
        allocationId: local.allocation,
        byteLength: { kind: "const", value: "24" },
        memorySpace: { kind: "global" },
        alignmentBytes: 4,
        aliasSetId: local.alias,
      }],
      indexMaps: [{
        indexMapId: local.indexMap,
        coordinateRank: 2,
        locationUnit: "element",
        location: {
          kind: "add",
          terms: [
            { kind: "mul", lhs: { kind: "coordinate", axis: 0 }, rhs: { kind: "const", value: "3" } },
            { kind: "coordinate", axis: 1 },
          ],
        },
        inBounds: {
          kind: "and",
          values: [
            { kind: "lessEqual", lhs: { kind: "const", value: "0" }, rhs: { kind: "coordinate", axis: 0 } },
            { kind: "lessEqual", lhs: { kind: "add", terms: [{ kind: "coordinate", axis: 0 }, { kind: "const", value: "1" }] }, rhs: { kind: "const", value: "2" } },
            { kind: "lessEqual", lhs: { kind: "const", value: "0" }, rhs: { kind: "coordinate", axis: 1 } },
            { kind: "lessEqual", lhs: { kind: "add", terms: [{ kind: "coordinate", axis: 1 }, { kind: "const", value: "1" }] }, rhs: { kind: "const", value: "3" } },
          ],
        },
      }],
      views: [{
        viewId: local.view,
        allocationId: local.allocation,
        dtype: "f32",
        byteOffset: { kind: "const", value: "0" },
        shape: [{ kind: "const", value: "2" }, { kind: "const", value: "3" }],
        indexMapId: local.indexMap,
        requiredAlignmentBytes: 4,
      }],
    },
  };
}

interface MutablePayload {
  symbols: Array<Record<string, unknown>>;
  constraints: Array<Record<string, unknown>>;
  allocations: Array<Record<string, unknown>>;
  indexMaps: Array<Record<string, unknown>>;
  views: Array<Record<string, unknown>>;
}

function mutablePayload(value: Record<string, unknown>): MutablePayload {
  return value.payload as unknown as MutablePayload;
}

async function diagnostic(run: () => Promise<unknown>): Promise<SemanticSchemaError> {
  try {
    await run();
    throw new Error("expected verification failure");
  } catch (error) {
    expect(error).toBeInstanceOf(SemanticSchemaError);
    return error as SemanticSchemaError;
  }
}

describe("verified layout artifacts", () => {
  it("verifies, freezes, remaps IDs, and hashes a normalized rank-2 artifact", async () => {
    const verified = await verifyLayoutArtifact(artifact());
    const payload = unwrapLayoutArtifact(verified);
    expect(Object.isFrozen(payload)).toBe(true);
    expect(Object.isFrozen(payload.views[0])).toBe(true);
    expect(payload.allocations[0]?.allocationId).toMatch(/^bg\.entity\.allocation\.scope-sha256\.[0-9a-f]{64}\.ordinal\.0$/u);
    expect(payload.indexMaps[0]?.indexMapId).toMatch(/^bg\.entity\.index-map\.scope-sha256\.[0-9a-f]{64}\.ordinal\.0$/u);
    expect(payload.views[0]?.allocationId).toBe(payload.allocations[0]?.allocationId);
    expect(await hashSemanticArtifact(verified)).toMatch(/^[0-9a-f]{64}$/u);
  });

  it("normalizes different raw local IDs to the same semantic hash", async () => {
    const first = await verifyLayoutArtifact(artifact());
    const second = await verifyLayoutArtifact(artifact({
      allocation: "renamedAllocation",
      alias: "renamedAlias",
      indexMap: "renamedMap",
      view: "renamedView",
    }));
    expect(await hashSemanticArtifact(first)).toBe(await hashSemanticArtifact(second));
  });

  it("decodes the byte entrypoint and excludes envelope provenance from hashes", async () => {
    const first = await decodeLayoutArtifact(new TextEncoder().encode(JSON.stringify(artifact())));
    const changed = artifact();
    changed.producer = { id: "other", version: "9" };
    changed.artifactId = "other-transport-id";
    changed.optionalMetadata = { note: "ignored by semantic hash" };
    const second = await decodeLayoutArtifact(new TextEncoder().encode(JSON.stringify(changed)));
    expect(await hashSemanticArtifact(first)).toBe(await hashSemanticArtifact(second));
  });

  it("rejects duplicate IDs, dangling references, unknown fields, rank drift, and bad alignment", async () => {
    const duplicate = artifact();
    const duplicatePayload = duplicate.payload as { allocations: unknown[] };
    duplicatePayload.allocations.push(structuredClone(duplicatePayload.allocations[0]));
    expect((await diagnostic(() => verifyLayoutArtifact(duplicate))).diagnostic.code)
      .toBe(LAYOUT_DIAGNOSTIC_CODES.duplicateId);

    const dangling = artifact();
    ((dangling.payload as { views: Array<{ allocationId: string }> }).views[0] as { allocationId: string }).allocationId = "missing";
    expect((await diagnostic(() => verifyLayoutArtifact(dangling))).diagnostic.code)
      .toBe(LAYOUT_DIAGNOSTIC_CODES.danglingReference);

    const unknown = artifact();
    ((unknown.payload as { views: Array<Record<string, unknown>> }).views[0] as Record<string, unknown>).newMeaning = true;
    expect((await diagnostic(() => verifyLayoutArtifact(unknown))).diagnostic.code)
      .toBe(LAYOUT_DIAGNOSTIC_CODES.unknownField);

    const rank = artifact();
    (rank.payload as { views: Array<{ shape: unknown[] }> }).views[0]?.shape.pop();
    expect((await diagnostic(() => verifyLayoutArtifact(rank))).diagnostic.code)
      .toBe(LAYOUT_DIAGNOSTIC_CODES.rankMismatch);

    const alignment = artifact();
    ((alignment.payload as { views: Array<{ requiredAlignmentBytes: number }> }).views[0] as { requiredAlignmentBytes: number }).requiredAlignmentBytes = 3;
    expect((await diagnostic(() => verifyLayoutArtifact(alignment))).diagnostic.code)
      .toBe(LAYOUT_DIAGNOSTIC_CODES.invalidAlignment);

    const offsetAlignment = artifact();
    (mutablePayload(offsetAlignment).views[0] as Record<string, unknown>).byteOffset = { kind: "const", value: "2" };
    expect((await diagnostic(() => verifyLayoutArtifact(offsetAlignment))).diagnostic.code)
      .toBe(LAYOUT_DIAGNOSTIC_CODES.invalidAlignment);
  });

  it("traces logical, predicate, and allocation bounds without clamping addresses", async () => {
    const verified = await verifyLayoutArtifact(artifact());
    const viewId = layoutArtifactPayload(verified).views[0]?.viewId;
    if (viewId === undefined) throw new Error("missing fixture view");
    const inBounds = traceViewCoordinate(verified, {
      viewId,
      coordinates: [parseWireI64("1"), parseWireI64("2")],
    });
    expect(inBounds).toMatchObject({
      mapLocation: { unit: "element", value: "5" },
      rootByteStart: "20",
      rootByteEndExclusive: "24",
      allocationByteLength: "24",
      logicalInBounds: true,
      predicateInBounds: true,
      allocationInBounds: true,
      accessInBounds: true,
    });

    const negative = traceViewCoordinate(verified, {
      viewId,
      coordinates: [parseWireI64("-1"), parseWireI64("0")],
    });
    expect(negative).toMatchObject({
      mapLocation: { value: "-3" },
      rootByteStart: "-12",
      logicalInBounds: false,
      predicateInBounds: false,
      allocationInBounds: false,
      accessInBounds: false,
    });
  });

  it("gives equivalent element and byte maps identical root addresses", async () => {
    const elementArtifact = await verifyLayoutArtifact(artifact());
    const byteValue = artifact();
    const byteMap = mutablePayload(byteValue).indexMaps[0];
    if (byteMap === undefined) throw new Error("missing fixture index map");
    byteMap.locationUnit = "byte";
    byteMap.location = {
      kind: "add",
      terms: [
        { kind: "mul", lhs: { kind: "coordinate", axis: 0 }, rhs: { kind: "const", value: "12" } },
        { kind: "mul", lhs: { kind: "coordinate", axis: 1 }, rhs: { kind: "const", value: "4" } },
      ],
    };
    const byteArtifact = await verifyLayoutArtifact(byteValue);
    const elementView = layoutArtifactPayload(elementArtifact).views[0]?.viewId;
    const byteView = layoutArtifactPayload(byteArtifact).views[0]?.viewId;
    if (elementView === undefined || byteView === undefined) throw new Error("missing fixture view");
    const coordinates = [parseWireI64("1"), parseWireI64("2")] as const;
    expect(traceViewCoordinate(elementArtifact, { viewId: elementView, coordinates }).rootByteStart)
      .toBe(traceViewCoordinate(byteArtifact, { viewId: byteView, coordinates }).rootByteStart);
  });

  it("separates root identity, alias-set possibility, and byte-range overlap", async () => {
    const sameRootValue = artifact();
    const sameRootPayload = mutablePayload(sameRootValue);
    const secondSameRootView = structuredClone(sameRootPayload.views[0]);
    if (secondSameRootView === undefined) throw new Error("missing fixture view");
    secondSameRootView.viewId = "view2";
    sameRootPayload.views.push(secondSameRootView);
    const sameRoot = await verifyLayoutArtifact(sameRootValue);
    const sameRootViews = layoutArtifactPayload(sameRoot).views;
    const sameRootTrace = traceViewAlias(sameRoot, {
      left: { viewId: sameRootViews[0]?.viewId ?? "", coordinates: [parseWireI64("0"), parseWireI64("0")] },
      right: { viewId: sameRootViews[1]?.viewId ?? "", coordinates: [parseWireI64("0"), parseWireI64("0")] },
    });
    expect(sameRootTrace).toMatchObject({
      sameAllocation: true,
      sameAliasSet: true,
      byteRangesOverlap: true,
      relation: "same-allocation",
    });

    const mayAliasValue = artifact();
    const mayAliasPayload = mutablePayload(mayAliasValue);
    const secondAllocation = structuredClone(mayAliasPayload.allocations[0]);
    const secondView = structuredClone(mayAliasPayload.views[0]);
    if (secondAllocation === undefined || secondView === undefined) throw new Error("missing fixture records");
    secondAllocation.allocationId = "alloc2";
    secondView.viewId = "view2";
    secondView.allocationId = "alloc2";
    mayAliasPayload.allocations.push(secondAllocation);
    mayAliasPayload.views.push(secondView);
    const mayAlias = await verifyLayoutArtifact(mayAliasValue);
    const mayAliasViews = layoutArtifactPayload(mayAlias).views;
    const mayAliasTrace = traceViewAlias(mayAlias, {
      left: { viewId: mayAliasViews[0]?.viewId ?? "", coordinates: [parseWireI64("0"), parseWireI64("0")] },
      right: { viewId: mayAliasViews[1]?.viewId ?? "", coordinates: [parseWireI64("0"), parseWireI64("0")] },
    });
    expect(mayAliasTrace).toMatchObject({
      sameAllocation: false,
      sameAliasSet: true,
      byteRangesOverlap: null,
      relation: "may-alias",
    });
  });

  it("requires complete in-domain dynamic bindings at trace time", async () => {
    const dynamicValue = artifact();
    const payload = mutablePayload(dynamicValue);
    payload.symbols.push({ id: "n", domain: { min: "1", max: "4" } });
    const allocation = payload.allocations[0];
    const view = payload.views[0];
    const indexMap = payload.indexMaps[0];
    if (allocation === undefined || view === undefined || indexMap === undefined) throw new Error("missing fixture records");
    allocation.byteLength = { kind: "mul", lhs: { kind: "symbol", id: "n" }, rhs: { kind: "const", value: "12" } };
    view.shape = [{ kind: "symbol", id: "n" }, { kind: "const", value: "3" }];
    const bounds = indexMap.inBounds as { values: Array<Record<string, unknown>> };
    (bounds.values[1] as { rhs: unknown }).rhs = { kind: "dimension", symbolId: "n" };
    const verified = await verifyLayoutArtifact(dynamicValue);
    const viewId = layoutArtifactPayload(verified).views[0]?.viewId ?? "";
    const coordinates = [parseWireI64("1"), parseWireI64("2")] as const;
    expect(() => traceViewCoordinate(verified, { viewId, coordinates }))
      .toThrowError(/BG-LAYOUT-UNRESOLVED-SYMBOL/u);
    expect(() => traceViewCoordinate(verified, {
      viewId,
      coordinates,
      bindings: { n: parseWireI64("5") },
    })).toThrowError(/BG-LAYOUT-SYMBOL-DOMAIN/u);
    expect(traceViewCoordinate(verified, {
      viewId,
      coordinates,
      bindings: { n: parseWireI64("2") },
    }).accessInBounds).toBe(true);
  });

  it("does not accept a structurally forged layout wrapper", () => {
    expect(() => unwrapLayoutArtifact(artifact() as unknown as VerifiedLayoutArtifact))
      .toThrowError(/BG-SCHEMA-UNVERIFIED-ARTIFACT/u);
  });
});
