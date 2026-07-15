import { describe, expect, it } from "vitest";
import {
  layoutArtifactPayload,
  traceViewCoordinate,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import { prepareViewCopyCpu } from "@unlocalhosted/browsergrad-semantic-core/kernel";
import { encodeWireI64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  VIEW_COPY_CONFORMANCE_CASE_IDS,
  cloneViewCopyConformanceWords,
  createViewCopyConformanceCases,
} from "../../../../test-support/view-copy-conformance-fixtures";

const EXACT_NAN_FILL = 0x7fc01234;

describe("shared view-copy conformance fixtures", () => {
  it("keeps a closed ordered artifact and complete-root contract", async () => {
    const cases = await createViewCopyConformanceCases();

    expect(Object.isFrozen(VIEW_COPY_CONFORMANCE_CASE_IDS)).toBe(true);
    expect(Object.isFrozen(cases)).toBe(true);
    expect(cases.map(({ id }) => id)).toEqual([...VIEW_COPY_CONFORMANCE_CASE_IDS]);
    expect(new Set(cases.map(({ id }) => id)).size).toBe(VIEW_COPY_CONFORMANCE_CASE_IDS.length);
    for (const fixture of cases) {
      const payload = layoutArtifactPayload(fixture.artifacts.layout);
      const sourceAllocation = payload.allocations.find(
        ({ allocationId }) => allocationId === fixture.artifacts.source.allocationId,
      );
      const destinationAllocation = payload.allocations.find(
        ({ allocationId }) => allocationId === fixture.artifacts.destination.allocationId,
      );
      if (sourceAllocation?.byteLength.kind !== "const" || destinationAllocation?.byteLength.kind !== "const") {
        throw new Error(`${fixture.id} must use static complete-root allocations`);
      }

      expect(fixture.layoutSemanticHash).toBe(fixture.artifacts.layoutSemanticHash);
      expect(fixture.kernelSemanticHash).toBe(fixture.artifacts.kernelSemanticHash);
      expect(fixture.layoutSemanticHash).toMatch(/^[0-9a-f]{64}$/u);
      expect(fixture.kernelSemanticHash).toMatch(/^[0-9a-f]{64}$/u);
      expect(fixture.sourceWords).toBeInstanceOf(Uint32Array);
      expect(fixture.expectedSourceWords).toBeInstanceOf(Uint32Array);
      expect(fixture.initialDestinationWords).toBeInstanceOf(Uint32Array);
      expect(fixture.expectedDestinationWords).toBeInstanceOf(Uint32Array);
      expect(fixture.sourceWords.byteLength).toBe(Number(sourceAllocation.byteLength.value));
      expect(fixture.expectedSourceWords.byteLength).toBe(fixture.sourceWords.byteLength);
      expect(fixture.initialDestinationWords.byteLength).toBe(Number(destinationAllocation.byteLength.value));
      expect(fixture.expectedDestinationWords.byteLength).toBe(fixture.initialDestinationWords.byteLength);
      expect(fixture.expectedSourceWords).toEqual(fixture.sourceWords);
      expect(fixture.expectedSourcePhysicalIndices).toHaveLength(fixture.expectedReadElements);
      expect(fixture.expectedDestinationPhysicalIndices).toHaveLength(
        fixture.expectedReadElements + fixture.expectedFilledElements,
      );
      expect(Object.isFrozen(fixture.expectedSourcePhysicalIndices)).toBe(true);
      expect(Object.isFrozen(fixture.expectedDestinationPhysicalIndices)).toBe(true);

      const cloned = cloneViewCopyConformanceWords(fixture.sourceWords);
      expect(cloned).toEqual(fixture.sourceWords);
      expect(cloned).not.toBe(fixture.sourceWords);
      cloned[0] = 0;
      expect(fixture.sourceWords[0]).not.toBe(0);

      const sourceWords = cloneViewCopyConformanceWords(fixture.sourceWords);
      const destinationWords = cloneViewCopyConformanceWords(fixture.initialDestinationWords);
      const prepared = await prepareViewCopyCpu(fixture.artifacts.layout, fixture.artifacts.kernel, {
        operationId: fixture.artifacts.operationId,
      });
      const execution = prepared.execute({
        source: bytes(sourceWords),
        destination: bytes(destinationWords),
      });

      expect(sourceWords).toEqual(fixture.expectedSourceWords);
      expect(destinationWords).toEqual(fixture.expectedDestinationWords);
      expect(execution.readElements).toBe(String(fixture.expectedReadElements));
      expect(execution.filledElements).toBe(String(fixture.expectedFilledElements));

      const coordinates = logicalCoordinates(fixture.logicalShape);
      const sourcePhysicalIndices = coordinates.flatMap((coordinate) => {
        const trace = traceViewCoordinate(fixture.artifacts.layout, {
          viewId: fixture.artifacts.source.viewId,
          coordinates: coordinate.map((value) => encodeWireI64(BigInt(value))),
        });
        return trace.accessInBounds ? [Number(BigInt(trace.rootByteStart) / 4n)] : [];
      });
      const destinationPhysicalIndices = coordinates.map((coordinate) => {
        const trace = traceViewCoordinate(fixture.artifacts.layout, {
          viewId: fixture.artifacts.destination.viewId,
          coordinates: coordinate.map((value) => encodeWireI64(BigInt(value))),
        });
        expect(trace.accessInBounds).toBe(true);
        return Number(BigInt(trace.rootByteStart) / 4n);
      });
      expect(sourcePhysicalIndices).toEqual(fixture.expectedSourcePhysicalIndices);
      expect(destinationPhysicalIndices).toEqual(fixture.expectedDestinationPhysicalIndices);
    }
  });

  it("preserves nonzero prefix/suffix canaries and exact NaN fill in padded roots", async () => {
    const paddingCases = (await createViewCopyConformanceCases()).filter(({ id }) => id.includes("padding"));
    expect(paddingCases.map(({ id }) => id)).toEqual([
      "rank2-padding-exact-nan",
      "rank3-padding-exact-nan",
    ]);

    for (const fixture of paddingCases) {
      const sourceLast = fixture.sourceWords.length - 1;
      const destinationLast = fixture.initialDestinationWords.length - 1;
      expect(fixture.sourceWords[0]).not.toBe(0);
      expect(fixture.sourceWords[sourceLast]).not.toBe(0);
      expect(fixture.initialDestinationWords[0]).not.toBe(0);
      expect(fixture.initialDestinationWords[destinationLast]).not.toBe(0);
      expect(fixture.expectedSourceWords[0]).toBe(fixture.sourceWords[0]);
      expect(fixture.expectedSourceWords[sourceLast]).toBe(fixture.sourceWords[sourceLast]);
      expect(fixture.expectedDestinationWords[0]).toBe(fixture.initialDestinationWords[0]);
      expect(fixture.expectedDestinationWords[destinationLast]).toBe(
        fixture.initialDestinationWords[destinationLast],
      );
      expect([...fixture.expectedDestinationWords].filter((word) => word === EXACT_NAN_FILL)).toHaveLength(
        fixture.expectedFilledElements,
      );
      expect(fixture.expectedSourcePhysicalIndices.every((index) => index > 0 && index < sourceLast)).toBe(true);
      expect(fixture.expectedDestinationPhysicalIndices.every(
        (index) => index > 0 && index < destinationLast,
      )).toBe(true);
    }
  });
});

function bytes(words: Uint32Array): Uint8Array {
  return new Uint8Array(words.buffer, words.byteOffset, words.byteLength);
}

function logicalCoordinates(shape: readonly number[]): readonly (readonly number[])[] {
  const coordinates: number[][] = [];
  const visit = (axis: number, prefix: readonly number[]) => {
    if (axis === shape.length) {
      coordinates.push([...prefix]);
      return;
    }
    for (let value = 0; value < shape[axis]!; value += 1) visit(axis + 1, [...prefix, value]);
  };
  visit(0, []);
  return coordinates;
}
