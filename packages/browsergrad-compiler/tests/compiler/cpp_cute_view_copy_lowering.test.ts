import {
  kernelArtifactPayload,
  prepareViewCopyCpu,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import { layoutArtifactPayload } from "@unlocalhosted/browsergrad-semantic-core/layout";
import { parseWireU64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { beforeAll, describe, expect, it } from "vitest";
import {
  lowerAuthorizedCppCuteViewCopyEntry,
  type LowerAuthorizedCppCuteViewCopyEntryRequest,
} from "../../src/index.js";
import type { AuthorizedCppCuteFrontendArtifact } from
  "../../src/cpp_cute_frontend_authorization.js";
import { unwrapVerifiedCppCuteFrontendArtifact } from
  "../../src/cpp_cute_frontend_artifact.js";
import type { CppCuteFrontendPayloadV3 } from "../../src/cpp_cute_frontend_types.js";
import {
  CPP_CUTE_FIXTURE_TEMPLATE_DECLARATION_ID,
} from "./support/cpp_cute_frontend_fixtures.js";
import {
  mutateCppCutePayloadToViewCopy,
} from "./support/cpp_cute_frontend_view_copy_fixtures.js";
import {
  createAuthorizedCppCuteProvenanceFixture,
  type AuthorizedCppCuteProvenanceFixture,
} from "./support/cpp_cute_provenance_fixtures.js";

const wire = (value: number) => parseWireU64(String(value));

let fixture: AuthorizedCppCuteProvenanceFixture;
let entryId: string;

beforeAll(async () => {
  fixture = await createAuthorizedCppCuteProvenanceFixture({
    mutatePayload: mutateCppCutePayloadToViewCopy,
  });
  const payload = unwrapVerifiedCppCuteFrontendArtifact(fixture.artifact).envelope.payload;
  if (payload.outcome.kind !== "accepted" || payload.outcome.selectedEntryIds[0] === undefined) {
    throw new Error("fixture lost selected view-copy entry");
  }
  entryId = payload.outcome.selectedEntryIds[0];
});

function request(overrides: Partial<LowerAuthorizedCppCuteViewCopyEntryRequest> = {}) {
  return {
    entryId,
    sourceAllocationByteLength: wire(32),
    destinationAllocationByteLength: wire(32),
    sourceByteOffset: wire(4),
    destinationByteOffset: wire(4),
    ...overrides,
  };
}

describe("authorized C++/CuTe view-copy lowering", () => {
  it("constructs canonical f32 semantic artifacts and executes the rank-2 transpose on CPU", async () => {
    const artifacts = await lowerAuthorizedCppCuteViewCopyEntry(fixture.authorization, request());
    const layout = layoutArtifactPayload(artifacts.layout);
    const kernel = kernelArtifactPayload(artifacts.kernel);
    const operation = kernel.operations[0];
    const cpu = await prepareViewCopyCpu(artifacts.layout, artifacts.kernel, {
      operationId: artifacts.operationId,
    });
    const sourceWords = new Uint32Array(8);
    sourceWords.set([0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000], 1);
    const destinationWords = new Uint32Array(8);
    destinationWords.fill(0xdeadbeef);
    const trace = cpu.execute({
      source: new Uint8Array(sourceWords.buffer),
      destination: new Uint8Array(destinationWords.buffer),
    });

    expect(artifacts.layoutSemanticHash).toBe(
      "5ade6e063773ba40a1046423e76776cf963544a26c7f17b301565d54a86ecdfe",
    );
    expect(artifacts.kernelSemanticHash).toBe(
      "64dc9d67e4f0de9c1f7b68fa369957c9521d8bbb9aa9725ac82f0dfaa573f409",
    );
    expect(artifacts.source).toEqual({
      allocationId: "bg.entity.allocation.scope-sha256.5361780f48f83d4ee2954aaa99251c2bdeb727c2fdc5a3afb676163ef709d509.ordinal.0",
      indexMapId: "bg.entity.index-map.scope-sha256.5361780f48f83d4ee2954aaa99251c2bdeb727c2fdc5a3afb676163ef709d509.ordinal.0",
      viewId: "bg.entity.view.scope-sha256.5361780f48f83d4ee2954aaa99251c2bdeb727c2fdc5a3afb676163ef709d509.ordinal.0",
    });
    expect(artifacts.destination).toEqual({
      allocationId: "bg.entity.allocation.scope-sha256.5361780f48f83d4ee2954aaa99251c2bdeb727c2fdc5a3afb676163ef709d509.ordinal.1",
      indexMapId: "bg.entity.index-map.scope-sha256.5361780f48f83d4ee2954aaa99251c2bdeb727c2fdc5a3afb676163ef709d509.ordinal.1",
      viewId: "bg.entity.view.scope-sha256.5361780f48f83d4ee2954aaa99251c2bdeb727c2fdc5a3afb676163ef709d509.ordinal.1",
    });
    expect(layout.allocations.map((allocation) => allocation.byteLength)).toEqual([
      { kind: "const", value: "32" },
      { kind: "const", value: "32" },
    ]);
    expect(layout.views.map((view) => view.byteOffset)).toEqual([
      { kind: "const", value: "4" },
      { kind: "const", value: "4" },
    ]);
    expect(operation).toMatchObject({
      kind: "view-copy",
      dtype: "f32",
      source: { access: "read", invalidSource: { kind: "reject" } },
      destination: { access: "write" },
      overlap: { kind: "forbid" },
    });
    expect([...destinationWords]).toEqual([
      0xdeadbeef,
      0x3f800000,
      0x40800000,
      0x40000000,
      0x40a00000,
      0x40400000,
      0x40c00000,
      0xdeadbeef,
    ]);
    expect(trace).toMatchObject({ readElements: "6", bytesWritten: "24", filledElements: "0" });
    expect(Object.isFrozen(artifacts)).toBe(true);
  });

  it("keeps authorization opaque and refuses caller semantic IDs or implicit storage capacity", async () => {
    await expect(lowerAuthorizedCppCuteViewCopyEntry(
      { ...fixture.authorization } as AuthorizedCppCuteFrontendArtifact,
      request(),
    )).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-AUTHORIZATION-UNVERIFIED" });
    await expect(lowerAuthorizedCppCuteViewCopyEntry(fixture.authorization, {
      ...request(),
      operationId: "caller-owned",
    } as never)).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-VIEW-COPY-INVALID-REQUEST",
      path: "$.request",
    });
    await expect(lowerAuthorizedCppCuteViewCopyEntry(fixture.authorization, request({
      sourceAllocationByteLength: wire(24),
    }))).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-LAYOUT",
    });
  });

  it("rejects non-f32, nullable, and asynchronous copy profiles", async () => {
    const nonF32 = await viewCopyFixture((payload) => {
      const tensor = payload.facts.find((fact) => fact.kind === "tensor");
      const elementType = tensor?.kind === "tensor"
        ? payload.types.find((type) => type.typeId === tensor.elementTypeId)
        : undefined;
      if (elementType?.kind !== "builtin") throw new Error("fixture lost element type");
      (elementType as { builtin: string }).builtin = "double";
    });
    await expect(lowerAuthorizedCppCuteViewCopyEntry(nonF32.authorization, requestFor(nonF32)))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-PROFILE",
        path: "$.artifact.source.elementTypeId",
      });

    const nullable = await viewCopyFixture((payload) => {
      const source = payload.facts.find((fact) => fact.kind === "tensor");
      if (source?.kind !== "tensor" || source.engine.kind !== "global-pointer") {
        throw new Error("fixture lost source engine");
      }
      (source.engine as { nullable: boolean }).nullable = true;
    });
    await expect(lowerAuthorizedCppCuteViewCopyEntry(nullable.authorization, requestFor(nullable)))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-PROFILE",
        path: "$.artifact.entry",
      });

    const asynchronous = await viewCopyFixture((payload) => {
      const intrinsic = payload.facts.find((fact) => fact.kind === "target-intrinsic");
      if (intrinsic?.kind !== "target-intrinsic" || intrinsic.operation.kind !== "copy") {
        throw new Error("fixture lost copy intrinsic");
      }
      (intrinsic.operation as { asynchronous: boolean }).asynchronous = true;
    });
    await expect(lowerAuthorizedCppCuteViewCopyEntry(asynchronous.authorization, requestFor(asynchronous)))
      .rejects.toMatchObject({
        code: "BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-PROFILE",
        path: "$.artifact.entry.operationExpressionId",
      });
  });

  it("rejects dynamic and non-positive affine layouts", async () => {
    const dynamicStride = await viewCopyFixture((payload) => {
      for (const layout of payload.facts.filter((fact) => fact.kind === "affine-layout")) {
        if (layout.kind !== "affine-layout" || layout.stride.kind !== "tuple") continue;
        const first = layout.stride.elements[0];
        if (first?.kind !== "scalar") throw new Error("fixture lost flat layout stride");
        (first as { value: unknown }).value = {
          kind: "runtime",
          declarationId: CPP_CUTE_FIXTURE_TEMPLATE_DECLARATION_ID,
        };
      }
    });
    await expect(lowerAuthorizedCppCuteViewCopyEntry(dynamicStride.authorization, requestFor(dynamicStride)))
      .rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-LAYOUT" });

    const negativeStride = await viewCopyFixture((payload) => {
      const sourceTensor = payload.facts.find((fact) => fact.kind === "tensor");
      const layout = sourceTensor?.kind === "tensor"
        ? payload.facts.find((fact) => fact.factId === sourceTensor.layoutFactId)
        : undefined;
      if (layout?.kind !== "affine-layout" || layout.stride.kind !== "tuple") {
        throw new Error("fixture lost source layout");
      }
      const first = layout.stride.elements[0];
      if (first?.kind !== "scalar") throw new Error("fixture lost flat layout stride");
      (first as { value: unknown }).value = { kind: "integer", value: "-1" };
      (layout as { cosize: unknown }).cosize = { kind: "integer", value: "2" };
    });
    await expect(lowerAuthorizedCppCuteViewCopyEntry(negativeStride.authorization, requestFor(negativeStride)))
      .rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-LAYOUT" });
  });

  it("fails closed for cancellation, accessors, malformed wire values, and limits", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(lowerAuthorizedCppCuteViewCopyEntry(fixture.authorization, request(), {
      signal: controller.signal,
    })).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-VIEW-COPY-CANCELLED",
      path: "$.options.signal",
    });
    const hostile: Record<string, unknown> = { ...request() };
    Object.defineProperty(hostile, "entryId", {
      enumerable: true,
      get: () => { throw new Error("must not execute"); },
    });
    await expect(lowerAuthorizedCppCuteViewCopyEntry(fixture.authorization, hostile as never))
      .rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-VIEW-COPY-INVALID-REQUEST" });
    await expect(lowerAuthorizedCppCuteViewCopyEntry(fixture.authorization, request({
      sourceByteOffset: "01" as never,
    }))).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-VIEW-COPY-INVALID-REQUEST" });
    await expect(lowerAuthorizedCppCuteViewCopyEntry(fixture.authorization, request({
      sourceByteOffset: wire(2),
    }))).rejects.toMatchObject({
      code: "BG-COMPILER-CPP-CUTE-VIEW-COPY-INVALID-REQUEST",
      path: "$.request.sourceByteOffset",
    });
    await expect(lowerAuthorizedCppCuteViewCopyEntry(fixture.authorization, request(), {
      limits: { maxNodes: 4 },
    })).rejects.toMatchObject({ code: "BG-COMPILER-CPP-CUTE-VIEW-COPY-RESOURCE-LIMIT" });
  });
});

async function viewCopyFixture(
  mutate: (payload: CppCuteFrontendPayloadV3) => void,
): Promise<AuthorizedCppCuteProvenanceFixture> {
  return createAuthorizedCppCuteProvenanceFixture({
    mutatePayload: async (payload) => {
      await mutateCppCutePayloadToViewCopy(payload);
      mutate(payload);
    },
  });
}

function requestFor(candidate: AuthorizedCppCuteProvenanceFixture) {
  const payload = unwrapVerifiedCppCuteFrontendArtifact(candidate.artifact).envelope.payload;
  if (payload.outcome.kind !== "accepted" || payload.outcome.selectedEntryIds[0] === undefined) {
    throw new Error("fixture lost selected entry");
  }
  return { ...request(), entryId: payload.outcome.selectedEntryIds[0] };
}
