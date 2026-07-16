import { layoutArtifactPayload } from "@unlocalhosted/browsergrad-semantic-core/layout";
import { parseWireI64, type WireI64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { beforeAll, describe, expect, it } from "vitest";
import { unwrapVerifiedCppCuteFrontendArtifact } from "../../src/cpp_cute_frontend_artifact.js";
import {
  lowerAuthorizedCppCuteLayoutEntry,
  traceLoweredCppCuteLayoutCoordinate,
  unwrapLoweredCppCuteLayoutEntry,
  type LoweredCppCuteLayoutEntry,
} from "../../src/cpp_cute_layout_lowering.js";
import type {
  AuthorizedCppCuteFrontendArtifact,
} from "../../src/cpp_cute_frontend_provenance.js";
import type {
  CppCuteAffineLayoutFactV1,
  CppCuteFrontendPayloadV1,
} from "../../src/cpp_cute_frontend_types.js";
import {
  CPP_CUTE_FIXTURE_ENTRY_ID,
  CPP_CUTE_FIXTURE_INTRINSIC_FACT_ID,
  CPP_CUTE_FIXTURE_LAYOUT_FACT_ID,
  CPP_CUTE_FIXTURE_TEMPLATE_DECLARATION_ID,
} from "./support/cpp_cute_frontend_fixtures.js";
import {
  createAuthorizedCppCuteProvenanceFixture,
  type AuthorizedCppCuteProvenanceFixture,
} from "./support/cpp_cute_provenance_fixtures.js";

const wire = (value: string): WireI64 => parseWireI64(value);

let canonicalFixture: AuthorizedCppCuteProvenanceFixture;

beforeAll(async () => {
  canonicalFixture = await createAuthorizedCppCuteProvenanceFixture();
});

async function lower(
  authorization: AuthorizedCppCuteFrontendArtifact = canonicalFixture.authorization,
  entryId = CPP_CUTE_FIXTURE_ENTRY_ID,
) {
  return lowerAuthorizedCppCuteLayoutEntry(authorization, { entryId });
}

function trace(lowered: LoweredCppCuteLayoutEntry, coordinates: readonly string[]) {
  return traceLoweredCppCuteLayoutCoordinate(lowered, { coordinates: coordinates.map(wire) });
}

function mutableLayoutFact(payload: CppCuteFrontendPayloadV1): Record<string, unknown> {
  const fact = payload.facts.find((candidate) => candidate.kind === "affine-layout");
  if (fact === undefined) throw new Error("fixture lost affine layout fact");
  return fact as unknown as Record<string, unknown>;
}

function integer(value: string) {
  return { kind: "integer", value };
}

describe("authorized C++/CuTe layout lowering", () => {
  it("lowers the selected layout fact through allocation-free shared semantics", async () => {
    const lowered = await lower();
    const record = unwrapLoweredCppCuteLayoutEntry(lowered);
    const payload = layoutArtifactPayload(record.preparedLayout.artifact);
    const frontend = unwrapVerifiedCppCuteFrontendArtifact(canonicalFixture.artifact).envelope.payload;

    expect(lowered).toMatchObject({
      layoutSemanticHash: "4e1fa226641c8441f503aa754b5c6d5bedc2449d9beb8987a7fa0cd222ce0667",
      coordinateRank: 2,
    });
    expect(lowered.originHash).toMatch(/^[0-9a-f]{64}$/u);
    expect(Object.isFrozen(lowered)).toBe(true);
    expect(record.entry.entryId).toBe(CPP_CUTE_FIXTURE_ENTRY_ID);
    expect(record.fact.factId).toBe(record.entry.layoutFactId);
    expect(record.originSpanRecords).toHaveLength(1);
    expect(payload.allocations).toEqual([]);
    expect(payload.views).toEqual([]);
    expect(payload.indexMaps).toHaveLength(1);
    expect(frontend.facts).toContainEqual(expect.objectContaining({
      factId: CPP_CUTE_FIXTURE_INTRINSIC_FACT_ID,
      kind: "target-intrinsic",
      availability: expect.objectContaining({ kind: "recognized-unsupported" }),
    }));

    const coordinates = [
      ["0", "0"], ["0", "1"], ["1", "0"],
      ["1", "1"], ["2", "0"], ["2", "1"],
    ];
    expect(coordinates.map((coordinate) => trace(lowered, coordinate).mapLocation.value))
      .toEqual(["0", "3", "1", "4", "2", "5"]);
  });

  it("preserves hierarchical top-level modes through explicit coordinate composition", async () => {
    const fixture = await createAuthorizedCppCuteProvenanceFixture({
      mutatePayload: (payload) => {
        const fact = mutableLayoutFact(payload);
        fact["shape"] = {
          kind: "tuple",
          elements: [
            { kind: "scalar", value: integer("3") },
            {
              kind: "tuple",
              elements: [
                { kind: "scalar", value: integer("2") },
                { kind: "scalar", value: integer("3") },
              ],
            },
          ],
        };
        fact["stride"] = {
          kind: "tuple",
          elements: [
            { kind: "scalar", value: integer("3") },
            {
              kind: "tuple",
              elements: [
                { kind: "scalar", value: integer("12") },
                { kind: "scalar", value: integer("1") },
              ],
            },
          ],
        };
        fact["rank"] = 2;
        fact["leafRank"] = 3;
        fact["size"] = integer("18");
        fact["cosize"] = integer("21");
      },
    });
    const lowered = await lower(fixture.authorization);
    expect(trace(lowered, ["1", "5"])).toMatchObject({
      logicalShape: ["3", "6"],
      mapLocation: { unit: "element", value: "17" },
      layoutInBounds: true,
    });
    const actual: string[] = [];
    for (let first = 0; first < 3; first += 1) {
      for (let second = 0; second < 6; second += 1) {
        actual.push(trace(lowered, [String(first), String(second)]).mapLocation.value);
      }
    }
    expect(actual).toEqual([
      "0", "12", "1", "13", "2", "14",
      "3", "15", "4", "16", "5", "17",
      "6", "18", "7", "19", "8", "20",
    ]);
  });

  it("keeps signed layout locations exact and validates CuTe cosize, not address span", async () => {
    const fixture = await createAuthorizedCppCuteProvenanceFixture({
      mutatePayload: (payload) => {
        const fact = mutableLayoutFact(payload);
        fact["stride"] = {
          kind: "tuple",
          elements: [
            { kind: "scalar", value: integer("-1") },
            { kind: "scalar", value: integer("3") },
          ],
        };
        fact["cosize"] = integer("2");
      },
    });
    const lowered = await lower(fixture.authorization);
    const coordinates = [
      ["0", "0"], ["0", "1"], ["1", "0"],
      ["1", "1"], ["2", "0"], ["2", "1"],
    ];
    expect(coordinates.map((coordinate) => trace(lowered, coordinate).mapLocation.value))
      .toEqual(["0", "3", "-1", "2", "-2", "1"]);
  });

  it("folds bounded static integer algebra to provenance-neutral layout identity", async () => {
    const fixture = await createAuthorizedCppCuteProvenanceFixture({
      mutatePayload: (payload) => {
        const fact = mutableLayoutFact(payload);
        const shape = fact["shape"] as { elements: Array<{ value: unknown }> };
        const stride = fact["stride"] as { elements: Array<{ value: unknown }> };
        if (shape.elements[0] === undefined || stride.elements[1] === undefined) throw new Error("fixture hierarchy drifted");
        shape.elements[0].value = { kind: "add", values: [integer("1"), integer("2")] };
        stride.elements[1].value = { kind: "multiply", values: [integer("1"), integer("3")] };
      },
    });
    const canonical = await lower();
    const equivalent = await lower(fixture.authorization);
    expect(equivalent.layoutSemanticHash).toBe(canonical.layoutSemanticHash);
    expect(equivalent.indexMapId).toBe(canonical.indexMapId);
    expect(equivalent.originHash).not.toBe(canonical.originHash);
  });

  it("rejects dynamic integers at every initial static-layout position", async () => {
    const cases: Array<(fact: Record<string, unknown>) => void> = [
      (fact) => {
        const shape = fact["shape"] as { elements: Array<{ value: unknown }> };
        if (shape.elements[0] === undefined) throw new Error("fixture shape drifted");
        shape.elements[0].value = { kind: "runtime", declarationId: CPP_CUTE_FIXTURE_TEMPLATE_DECLARATION_ID };
      },
      (fact) => {
        const stride = fact["stride"] as { elements: Array<{ value: unknown }> };
        if (stride.elements[0] === undefined) throw new Error("fixture stride drifted");
        stride.elements[0].value = { kind: "runtime", declarationId: CPP_CUTE_FIXTURE_TEMPLATE_DECLARATION_ID };
      },
      (fact) => { fact["size"] = { kind: "runtime", declarationId: CPP_CUTE_FIXTURE_TEMPLATE_DECLARATION_ID }; },
      (fact) => { fact["cosize"] = { kind: "runtime", declarationId: CPP_CUTE_FIXTURE_TEMPLATE_DECLARATION_ID }; },
    ];
    for (const mutate of cases) {
      const fixture = await createAuthorizedCppCuteProvenanceFixture({
        mutatePayload: (payload) => mutate(mutableLayoutFact(payload)),
      });
      await expect(lower(fixture.authorization)).rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-LAYOUT-UNSUPPORTED-LAYOUT",
      });
    }
  });

  it("requires one explicit selected layout entry", async () => {
    await expect(lower(canonicalFixture.authorization, `bg.cpp.entry.sha256.${"0".repeat(64)}`)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-LAYOUT-UNSUPPORTED-ENTRY",
      path: "$.artifact.outcome.selectedEntryIds",
    });

    const extraEntryId = `bg.cpp.entry.sha256.${"f".repeat(64)}`;
    await expect(createAuthorizedCppCuteProvenanceFixture({
      mutatePayload: (payload) => {
        const entries = payload.entries as CppCuteFrontendPayloadV1["entries"] & Array<Record<string, unknown>>;
        entries.push({
          entryId: extraEntryId,
          kind: "layout",
          layoutFactId: CPP_CUTE_FIXTURE_LAYOUT_FACT_ID,
          selectedRootDeclarationIds: [
            (payload.facts.find((fact) => fact.kind === "affine-layout") as CppCuteAffineLayoutFactV1).resultDeclarationId,
          ],
        });
        (payload.outcome as unknown as { selectedEntryIds: string[] }).selectedEntryIds.push(extraEntryId);
      },
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-AOT-RECEIPT-OUTPUT-MISMATCH",
      path: "$.artifact.payload.outcome",
    });
  });

  it("rejects ambiguous result ownership even after structural verification", async () => {
    const fixture = await createAuthorizedCppCuteProvenanceFixture({
      mutatePayload: (payload) => {
        const original = payload.facts.find((fact) => fact.kind === "affine-layout") as CppCuteAffineLayoutFactV1;
        (payload.facts as CppCuteFrontendPayloadV1["facts"] & CppCuteAffineLayoutFactV1[]).push({
          ...structuredClone(original),
          factId: `bg.cpp.fact.sha256.${"f".repeat(64)}`,
        });
      },
    });
    await expect(lower(fixture.authorization)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-LAYOUT-INCONSISTENT-ARTIFACT",
      path: "$.artifact.facts",
    });
  });

  it("keeps frontend authorization and lowered authority instance-bound", async () => {
    await expect(lower({ ...canonicalFixture.authorization } as AuthorizedCppCuteFrontendArtifact)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-PROVENANCE-UNVERIFIED",
    });
    const lowered = await lower();
    expect(() => trace({ ...lowered } as LoweredCppCuteLayoutEntry, ["0", "0"]))
      .toThrowError(expect.objectContaining({ code: "BG-COMPILER-CPP-CUTE-LAYOUT-UNVERIFIED" }));
  });

  it("fails closed on cancellation, open requests, hostile options, and budgets", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(lowerAuthorizedCppCuteLayoutEntry(canonicalFixture.authorization, {
      entryId: CPP_CUTE_FIXTURE_ENTRY_ID,
    }, { signal: controller.signal })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-LAYOUT-CANCELLED",
    });
    await expect(lowerAuthorizedCppCuteLayoutEntry(canonicalFixture.authorization, {
      entryId: CPP_CUTE_FIXTURE_ENTRY_ID,
      extra: true,
    } as never)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-LAYOUT-INVALID-REQUEST",
    });
    const hostile: Record<string, unknown> = {};
    Object.defineProperty(hostile, "limits", { enumerable: true, get: () => ({ maxNodes: 10 }) });
    await expect(lowerAuthorizedCppCuteLayoutEntry(canonicalFixture.authorization, {
      entryId: CPP_CUTE_FIXTURE_ENTRY_ID,
    }, hostile)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-LAYOUT-INVALID-REQUEST",
    });
    await expect(lowerAuthorizedCppCuteLayoutEntry(canonicalFixture.authorization, {
      entryId: CPP_CUTE_FIXTURE_ENTRY_ID,
    }, { limits: { maxNodes: 4 } })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-LAYOUT-RESOURCE-LIMIT",
    });
    await expect(lowerAuthorizedCppCuteLayoutEntry(canonicalFixture.authorization, {
      entryId: CPP_CUTE_FIXTURE_ENTRY_ID,
    }, { limits: { maxIntegerBits: 2 } })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-LAYOUT-RESOURCE-LIMIT",
    });
  });
});
