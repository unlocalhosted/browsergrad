import { describe, expect, it } from "vitest";

import {
  normalizeLayoutExpr,
  type DimExpr,
  type IndexExpr,
  type LayoutExpr,
  type PredicateExpr,
} from "../../src/layout";
import { evaluateIndexExpr, evaluatePredicateExpr } from "../../src/layout/index-eval";
import { encodeWireI64 } from "../../src/schema";

const constant = (value: bigint): DimExpr => ({ kind: "const", value: encodeWireI64(value) });
const coordinate = (axis: number): IndexExpr => ({ kind: "coordinate", axis });

function strided(shape: readonly bigint[], strides: readonly bigint[]): LayoutExpr {
  return { kind: "strided", shape: shape.map(constant), strides: strides.map(constant) };
}

function rowMajorStrides(shape: readonly bigint[]): readonly bigint[] {
  const result = Array.from<bigint>({ length: shape.length }).fill(1n);
  for (let axis = shape.length - 2; axis >= 0; axis -= 1) {
    result[axis] = (result[axis + 1] as bigint) * (shape[axis + 1] as bigint);
  }
  return result;
}

function address(expression: IndexExpr, rank: number, coordinates: readonly bigint[]): bigint {
  const result = evaluateIndexExpr(expression, { coordinateRank: rank, coordinates });
  if (result.kind !== "resolved") throw new Error("generated constant layout unexpectedly unresolved");
  return result.value;
}

function inBounds(expression: PredicateExpr, rank: number, coordinates: readonly bigint[]): boolean {
  const result = evaluatePredicateExpr(expression, { coordinateRank: rank, coordinates });
  if (result.kind !== "resolved") throw new Error("generated constant predicate unexpectedly unresolved");
  return result.value;
}

function linear(coordinates: readonly bigint[], strides: readonly bigint[]): bigint {
  return coordinates.reduce((sum, coordinate, axis) => sum + coordinate * (strides[axis] as bigint), 0n);
}

function generator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state;
  };
}

describe("deterministic generated layout properties", () => {
  it("preserves address and bounds semantics across strided transforms", () => {
    const next = generator(0x5eedc0de);
    for (let sample = 0; sample < 96; sample += 1) {
      const rank = 1 + (next() % 4);
      const shape = Array.from({ length: rank }, () => BigInt(1 + (next() % 5)));
      const strides = rowMajorStrides(shape);
      const sourceCoordinates = shape.map((extent) => BigInt(next() % Number(extent)));
      const expected = linear(sourceCoordinates, strides);

      const base = normalizeLayoutExpr(strided(shape, strides));
      expect(address(base.location, rank, sourceCoordinates)).toBe(expected);
      expect(inBounds(base.inBounds, rank, sourceCoordinates)).toBe(true);
      for (let axis = 0; axis < rank; axis += 1) {
        const below = [...sourceCoordinates];
        below[axis] = -1n;
        const above = [...sourceCoordinates];
        above[axis] = shape[axis] as bigint;
        expect(inBounds(base.inBounds, rank, below)).toBe(false);
        expect(inBounds(base.inBounds, rank, above)).toBe(false);
      }

      const signedNonContiguousStrides = strides.map((stride, axis) => (
        axis % 2 === 0 ? stride * 2n : -stride
      ));
      const signedNonContiguous = normalizeLayoutExpr(strided(shape, signedNonContiguousStrides));
      expect(address(signedNonContiguous.location, rank, sourceCoordinates))
        .toBe(linear(sourceCoordinates, signedNonContiguousStrides));
      expect(inBounds(signedNonContiguous.inBounds, rank, sourceCoordinates)).toBe(true);

      const axes = Array.from({ length: rank }, (_, axis) => rank - axis - 1);
      const permuted = normalizeLayoutExpr({ kind: "permute", source: strided(shape, strides), axes });
      const permutedCoordinates = [...sourceCoordinates].reverse();
      expect(address(permuted.location, rank, permutedCoordinates)).toBe(expected);
      expect(inBounds(permuted.inBounds, rank, permutedCoordinates)).toBe(true);

      const steps = shape.map((extent) => extent > 1n && next() % 2 === 0 ? 2n : 1n);
      const offsets = shape.map((extent, axis) => {
        const step = steps[axis] as bigint;
        return BigInt(next() % Number(extent - (step > 1n ? 1n : 0n)));
      });
      const sizes = shape.map((extent, axis) => ((extent - 1n - (offsets[axis] as bigint)) / (steps[axis] as bigint)) + 1n);
      const sliceCoordinates = sizes.map((extent) => BigInt(next() % Number(extent)));
      const slicedSourceCoordinates = sliceCoordinates.map((coordinate, axis) => (
        (offsets[axis] as bigint) + coordinate * (steps[axis] as bigint)
      ));
      const sliced = normalizeLayoutExpr({
        kind: "slice",
        source: strided(shape, strides),
        offsets: offsets.map(constant),
        sizes: sizes.map(constant),
        steps: steps.map(constant),
      });
      expect(address(sliced.location, rank, sliceCoordinates)).toBe(linear(slicedSourceCoordinates, strides));
      expect(inBounds(sliced.inBounds, rank, sliceCoordinates)).toBe(true);
      const slicePastEnd = [...sliceCoordinates];
      slicePastEnd[0] = sizes[0] as bigint;
      expect(inBounds(sliced.inBounds, rank, slicePastEnd)).toBe(false);

      const reverseCoordinates = shape.map((extent) => BigInt(next() % Number(extent)));
      const reversedSourceCoordinates = reverseCoordinates.map((coordinateValue, axis) => (
        (shape[axis] as bigint) - 1n - coordinateValue
      ));
      const reversed = normalizeLayoutExpr({
        kind: "slice",
        source: strided(shape, strides),
        offsets: shape.map((extent) => constant(extent - 1n)),
        sizes: shape.map(constant),
        steps: shape.map(() => constant(-1n)),
      });
      expect(address(reversed.location, rank, reverseCoordinates)).toBe(linear(reversedSourceCoordinates, strides));
      expect(inBounds(reversed.inBounds, rank, reverseCoordinates)).toBe(true);

      const composed = normalizeLayoutExpr({
        kind: "compose",
        source: strided(shape, strides),
        shape: [...shape].reverse().map(constant),
        sourceCoordinates: Array.from({ length: rank }, (_, axis) => coordinate(rank - axis - 1)),
      });
      expect(address(composed.location, rank, permutedCoordinates)).toBe(expected);
      expect(inBounds(composed.inBounds, rank, permutedCoordinates)).toBe(true);

      const singletonShape = [1n, ...shape.slice(1)];
      const singletonStrides = rowMajorStrides(singletonShape);
      const broadcastShape = [BigInt(2 + (next() % 4)), ...shape.slice(1)];
      const broadcastCoordinates = [BigInt(next() % Number(broadcastShape[0])), ...sourceCoordinates.slice(1)];
      const broadcast = normalizeLayoutExpr({
        kind: "broadcast",
        source: strided(singletonShape, singletonStrides),
        shape: broadcastShape.map(constant),
      });
      expect(address(broadcast.location, rank, broadcastCoordinates))
        .toBe(linear([0n, ...broadcastCoordinates.slice(1)], singletonStrides));
      expect(inBounds(broadcast.inBounds, rank, broadcastCoordinates)).toBe(true);

      const leadingExtent = BigInt(2 + (next() % 4));
      const leadingBroadcast = normalizeLayoutExpr({
        kind: "broadcast",
        source: strided(shape, strides),
        shape: [constant(leadingExtent), ...shape.map(constant)],
      });
      const leadingCoordinates = [BigInt(next() % Number(leadingExtent)), ...sourceCoordinates];
      expect(address(leadingBroadcast.location, rank + 1, leadingCoordinates)).toBe(expected);
      expect(inBounds(leadingBroadcast.inBounds, rank + 1, leadingCoordinates)).toBe(true);

      const low = shape.map(() => BigInt(next() % 3));
      const high = shape.map(() => BigInt(next() % 3));
      const paddedCoordinates = sourceCoordinates.map((coordinate, axis) => coordinate + (low[axis] as bigint));
      const padded = normalizeLayoutExpr({
        kind: "pad",
        source: strided(shape, strides),
        low: low.map(constant),
        high: high.map(constant),
      });
      expect(address(padded.location, rank, paddedCoordinates)).toBe(expected);
      expect(inBounds(padded.inBounds, rank, paddedCoordinates)).toBe(true);
      const halo = [...paddedCoordinates];
      halo[0] = (low[0] as bigint) - 1n;
      expect(inBounds(padded.inBounds, rank, halo)).toBe(false);
    }
  });
});
