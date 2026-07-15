import { LAYOUT_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import { canonicalizeJson } from "../schema/canonical-json.js";
import { deepFreezeJson, type JsonValue } from "../schema/json.js";
import { encodeWireI64, parseWireI64, wireIntegerToBigInt } from "../schema/integers.js";
import { type DecodeLimits, resolveDecodeLimits } from "../schema/limits.js";
import type { DimExpr } from "./dim-expr.js";
import type { IndexExpr, LayoutExpr, PredicateExpr } from "./model.js";

export interface NormalizedLayout {
  readonly shape: readonly DimExpr[];
  readonly coordinateRank: number;
  readonly locationUnit: "element";
  readonly location: IndexExpr;
  readonly inBounds: PredicateExpr;
}

interface NormalizationState {
  readonly limits: DecodeLimits;
  nodes: number;
}

export function normalizeLayoutExpr(
  expression: LayoutExpr,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): NormalizedLayout {
  const state: NormalizationState = {
    limits: resolveDecodeLimits(options.limits),
    nodes: 0,
  };
  const normalized = normalize(expression, state, "$", 1);
  return deepFreezeJson(normalized as unknown as JsonValue) as unknown as NormalizedLayout;
}

function normalize(
  expression: LayoutExpr,
  state: NormalizationState,
  path: string,
  depth: number,
): NormalizedLayout {
  consume(state, path, depth);
  switch (expression.kind) {
    case "strided": {
      requireRank(expression.shape, state, `${path}.shape`);
      if (expression.strides.length !== expression.shape.length) {
        invalid(`${path}.strides`, "strided layout shape and stride ranks must match");
      }
      const shape = expression.shape.map((value, axis) => cloneDimExpr(value, state, `${path}.shape[${axis}]`, depth + 1));
      requireNonnegativeConstants(shape, `${path}.shape`);
      const strides = expression.strides.map((value, axis) => cloneDimExpr(value, state, `${path}.strides[${axis}]`, depth + 1));
      const location = addExpr(shape.map((_, axis) => mulExpr(coordinate(axis), dimExprToIndexExpr(strides[axis] as DimExpr))));
      return layout(shape, location, logicalBounds(shape));
    }
    case "compose": {
      requireRank(expression.shape, state, `${path}.shape`);
      const source = normalize(expression.source, state, `${path}.source`, depth + 1);
      if (expression.sourceCoordinates.length !== source.coordinateRank) {
        invalid(`${path}.sourceCoordinates`, "composition must provide exactly one coordinate for every source axis");
      }
      const shape = expression.shape.map((value, axis) => cloneDimExpr(value, state, `${path}.shape[${axis}]`, depth + 1));
      requireNonnegativeConstants(shape, `${path}.shape`);
      const sourceCoordinates = expression.sourceCoordinates.map((value, axis) => (
        cloneIndexExpr(value, shape.length, state, `${path}.sourceCoordinates[${axis}]`, depth + 1)
      ));
      return layout(
        shape,
        substituteIndex(source.location, sourceCoordinates),
        andExpr([logicalBounds(shape), substitutePredicate(source.inBounds, sourceCoordinates)]),
      );
    }
    case "permute": {
      const source = normalize(expression.source, state, `${path}.source`, depth + 1);
      const axes = validatePermutation(expression.axes, source.coordinateRank, `${path}.axes`);
      const inverse = Array.from({ length: axes.length }, () => -1);
      for (const [outputAxis, sourceAxis] of axes.entries()) inverse[sourceAxis] = outputAxis;
      const sourceCoordinates = inverse.map((outputAxis) => coordinate(outputAxis));
      const shape = axes.map((sourceAxis) => source.shape[sourceAxis] as DimExpr);
      return layout(
        shape,
        substituteIndex(source.location, sourceCoordinates),
        andExpr([logicalBounds(shape), substitutePredicate(source.inBounds, sourceCoordinates)]),
      );
    }
    case "slice": {
      const source = normalize(expression.source, state, `${path}.source`, depth + 1);
      requireTransformRank(expression.offsets, source.coordinateRank, `${path}.offsets`);
      requireTransformRank(expression.sizes, source.coordinateRank, `${path}.sizes`);
      requireTransformRank(expression.steps, source.coordinateRank, `${path}.steps`);
      const offsets = expression.offsets.map((value, axis) => cloneDimExpr(value, state, `${path}.offsets[${axis}]`, depth + 1));
      const shape = expression.sizes.map((value, axis) => cloneDimExpr(value, state, `${path}.sizes[${axis}]`, depth + 1));
      const steps = expression.steps.map((value, axis) => cloneDimExpr(value, state, `${path}.steps[${axis}]`, depth + 1));
      requireNonnegativeConstants(shape, `${path}.sizes`);
      for (const [axis, step] of steps.entries()) {
        if (isConstant(step, 0n)) invalid(`${path}.steps[${axis}]`, "slice step must not be zero");
      }
      const sourceCoordinates = offsets.map((offset, axis) => (
        addExpr([dimExprToIndexExpr(offset), mulExpr(coordinate(axis), dimExprToIndexExpr(steps[axis] as DimExpr))])
      ));
      const nonZeroSteps = steps.map((step) => notExpr(equalExpr(dimExprToIndexExpr(step), integer(0n))));
      return layout(
        shape,
        substituteIndex(source.location, sourceCoordinates),
        andExpr([logicalBounds(shape), ...nonZeroSteps, substitutePredicate(source.inBounds, sourceCoordinates)]),
      );
    }
    case "broadcast": {
      const source = normalize(expression.source, state, `${path}.source`, depth + 1);
      requireRank(expression.shape, state, `${path}.shape`);
      const shape = expression.shape.map((value, axis) => cloneDimExpr(value, state, `${path}.shape[${axis}]`, depth + 1));
      requireNonnegativeConstants(shape, `${path}.shape`);
      if (shape.length < source.coordinateRank) invalid(`${path}.shape`, "broadcast target rank must be at least the source rank");
      const leading = shape.length - source.coordinateRank;
      const sourceCoordinates = source.shape.map((sourceDim, sourceAxis) => {
        const outputAxis = leading + sourceAxis;
        const outputDim = shape[outputAxis] as DimExpr;
        if (isConstant(sourceDim, 1n)) return integer(0n);
        if (!sameDimExpr(sourceDim, outputDim)) {
          invalid(`${path}.shape[${outputAxis}]`, "broadcast requires a source extent of one or a structurally equal target extent");
        }
        return coordinate(outputAxis);
      });
      return layout(
        shape,
        substituteIndex(source.location, sourceCoordinates),
        andExpr([logicalBounds(shape), substitutePredicate(source.inBounds, sourceCoordinates)]),
      );
    }
    case "pad": {
      const source = normalize(expression.source, state, `${path}.source`, depth + 1);
      requireTransformRank(expression.low, source.coordinateRank, `${path}.low`);
      requireTransformRank(expression.high, source.coordinateRank, `${path}.high`);
      const low = expression.low.map((value, axis) => cloneDimExpr(value, state, `${path}.low[${axis}]`, depth + 1));
      const high = expression.high.map((value, axis) => cloneDimExpr(value, state, `${path}.high[${axis}]`, depth + 1));
      requireNonnegativeConstants(low, `${path}.low`);
      requireNonnegativeConstants(high, `${path}.high`);
      const shape = source.shape.map((sourceDim, axis) => ({
        kind: "add" as const,
        terms: [low[axis] as DimExpr, sourceDim, high[axis] as DimExpr],
      }));
      const sourceCoordinates = low.map((value, axis) => (
        addExpr([coordinate(axis), mulExpr(integer(-1n), dimExprToIndexExpr(value))])
      ));
      const validPadding = [...low, ...high].map((value) => lessEqualExpr(integer(0n), dimExprToIndexExpr(value)));
      return layout(
        shape,
        substituteIndex(source.location, sourceCoordinates),
        andExpr([logicalBounds(shape), ...validPadding, substitutePredicate(source.inBounds, sourceCoordinates)]),
      );
    }
  }
}

function layout(shape: readonly DimExpr[], location: IndexExpr, inBounds: PredicateExpr): NormalizedLayout {
  return {
    shape,
    coordinateRank: shape.length,
    locationUnit: "element",
    location,
    inBounds,
  };
}

function logicalBounds(shape: readonly DimExpr[]): PredicateExpr {
  return andExpr(shape.flatMap((dimension, axis) => [
    lessEqualExpr(integer(0n), coordinate(axis)),
    lessEqualExpr(addExpr([coordinate(axis), integer(1n)]), dimExprToIndexExpr(dimension)),
  ]));
}

function dimExprToIndexExpr(expression: DimExpr): IndexExpr {
  switch (expression.kind) {
    case "const": return { kind: "const", value: expression.value };
    case "symbol": return { kind: "dimension", symbolId: expression.id };
    case "add": return addExpr(expression.terms.map(dimExprToIndexExpr));
    case "mul": return mulExpr(dimExprToIndexExpr(expression.lhs), dimExprToIndexExpr(expression.rhs));
    case "floorDiv":
    case "ceilDiv":
    case "mod":
      return {
        kind: expression.kind,
        value: dimExprToIndexExpr(expression.value),
        divisor: dimExprToIndexExpr(expression.divisor),
      };
    case "min":
    case "max":
      return { kind: expression.kind, values: expression.values.map(dimExprToIndexExpr) };
  }
}

function cloneDimExpr(
  expression: DimExpr,
  state: NormalizationState,
  path: string,
  depth: number,
): DimExpr {
  consume(state, path, depth);
  switch (expression.kind) {
    case "const": return { kind: "const", value: parseWireI64(expression.value, `${path}.value`) };
    case "symbol": {
      if (!/^[A-Za-z_][A-Za-z0-9_.-]*$/u.test(expression.id)) invalid(`${path}.id`, "invalid dimension symbol ID");
      return { kind: "symbol", id: expression.id };
    }
    case "add": {
      if (expression.terms.length === 0) invalid(`${path}.terms`, "add requires at least one term");
      return { kind: "add", terms: expression.terms.map((term, index) => cloneDimExpr(term, state, `${path}.terms[${index}]`, depth + 1)) };
    }
    case "mul": return {
      kind: "mul",
      lhs: cloneDimExpr(expression.lhs, state, `${path}.lhs`, depth + 1),
      rhs: cloneDimExpr(expression.rhs, state, `${path}.rhs`, depth + 1),
    };
    case "floorDiv":
    case "ceilDiv":
    case "mod":
      return {
        kind: expression.kind,
        value: cloneDimExpr(expression.value, state, `${path}.value`, depth + 1),
        divisor: cloneDimExpr(expression.divisor, state, `${path}.divisor`, depth + 1),
      };
    case "min":
    case "max": {
      if (expression.values.length === 0) invalid(`${path}.values`, `${expression.kind} requires at least one value`);
      return { kind: expression.kind, values: expression.values.map((value, index) => cloneDimExpr(value, state, `${path}.values[${index}]`, depth + 1)) };
    }
  }
}

function cloneIndexExpr(
  expression: IndexExpr,
  coordinateRank: number,
  state: NormalizationState,
  path: string,
  depth: number,
): IndexExpr {
  consume(state, path, depth);
  switch (expression.kind) {
    case "const": return { kind: "const", value: parseWireI64(expression.value, `${path}.value`) };
    case "coordinate": {
      if (!Number.isSafeInteger(expression.axis) || expression.axis < 0 || expression.axis >= coordinateRank) {
        invalid(`${path}.axis`, `coordinate axis must be in [0, ${coordinateRank})`);
      }
      return coordinate(expression.axis);
    }
    case "dimension": {
      if (!/^[A-Za-z_][A-Za-z0-9_.-]*$/u.test(expression.symbolId)) invalid(`${path}.symbolId`, "invalid dimension symbol ID");
      return { kind: "dimension", symbolId: expression.symbolId };
    }
    case "add": {
      if (expression.terms.length === 0) invalid(`${path}.terms`, "index add requires at least one term");
      return addExpr(expression.terms.map((term, index) => cloneIndexExpr(term, coordinateRank, state, `${path}.terms[${index}]`, depth + 1)));
    }
    case "mul": return mulExpr(
      cloneIndexExpr(expression.lhs, coordinateRank, state, `${path}.lhs`, depth + 1),
      cloneIndexExpr(expression.rhs, coordinateRank, state, `${path}.rhs`, depth + 1),
    );
    case "floorDiv":
    case "ceilDiv":
    case "mod":
      return {
        kind: expression.kind,
        value: cloneIndexExpr(expression.value, coordinateRank, state, `${path}.value`, depth + 1),
        divisor: cloneIndexExpr(expression.divisor, coordinateRank, state, `${path}.divisor`, depth + 1),
      };
    case "min":
    case "max": {
      if (expression.values.length === 0) invalid(`${path}.values`, `index ${expression.kind} requires at least one value`);
      return {
        kind: expression.kind,
        values: expression.values.map((value, index) => cloneIndexExpr(value, coordinateRank, state, `${path}.values[${index}]`, depth + 1)),
      };
    }
  }
}

function substituteIndex(expression: IndexExpr, coordinates: readonly IndexExpr[]): IndexExpr {
  switch (expression.kind) {
    case "coordinate": return coordinates[expression.axis] as IndexExpr;
    case "const":
    case "dimension": return expression;
    case "add": return addExpr(expression.terms.map((term) => substituteIndex(term, coordinates)));
    case "mul": return mulExpr(substituteIndex(expression.lhs, coordinates), substituteIndex(expression.rhs, coordinates));
    case "floorDiv":
    case "ceilDiv":
    case "mod": return {
      kind: expression.kind,
      value: substituteIndex(expression.value, coordinates),
      divisor: substituteIndex(expression.divisor, coordinates),
    };
    case "min":
    case "max": return { kind: expression.kind, values: expression.values.map((value) => substituteIndex(value, coordinates)) };
  }
}

function substitutePredicate(expression: PredicateExpr, coordinates: readonly IndexExpr[]): PredicateExpr {
  switch (expression.kind) {
    case "bool": return expression;
    case "equal": return equalExpr(substituteIndex(expression.lhs, coordinates), substituteIndex(expression.rhs, coordinates));
    case "lessEqual": return lessEqualExpr(substituteIndex(expression.lhs, coordinates), substituteIndex(expression.rhs, coordinates));
    case "and": return andExpr(expression.values.map((value) => substitutePredicate(value, coordinates)));
    case "or": return orExpr(expression.values.map((value) => substitutePredicate(value, coordinates)));
    case "not": return notExpr(substitutePredicate(expression.value, coordinates));
  }
}

function coordinate(axis: number): IndexExpr {
  return { kind: "coordinate", axis };
}

function integer(value: bigint): IndexExpr {
  return { kind: "const", value: encodeWireI64(value) };
}

function addExpr(values: readonly IndexExpr[]): IndexExpr {
  const terms = values.flatMap((value) => value.kind === "add" ? value.terms : [value])
    .filter((value) => !isIndexConstant(value, 0n));
  if (terms.length === 0) return integer(0n);
  if (terms.length === 1) return terms[0] as IndexExpr;
  return { kind: "add", terms };
}

function mulExpr(lhs: IndexExpr, rhs: IndexExpr): IndexExpr {
  if (isIndexConstant(lhs, 0n) || isIndexConstant(rhs, 0n)) return integer(0n);
  if (isIndexConstant(lhs, 1n)) return rhs;
  if (isIndexConstant(rhs, 1n)) return lhs;
  return { kind: "mul", lhs, rhs };
}

function equalExpr(lhs: IndexExpr, rhs: IndexExpr): PredicateExpr {
  return { kind: "equal", lhs, rhs };
}

function lessEqualExpr(lhs: IndexExpr, rhs: IndexExpr): PredicateExpr {
  return { kind: "lessEqual", lhs, rhs };
}

function andExpr(values: readonly PredicateExpr[]): PredicateExpr {
  const flattened = values.flatMap((value) => value.kind === "and" ? value.values : [value]);
  if (flattened.some((value) => value.kind === "bool" && !value.value)) return { kind: "bool", value: false };
  const meaningful = flattened.filter((value) => value.kind !== "bool" || !value.value);
  if (meaningful.length === 0) return { kind: "bool", value: true };
  if (meaningful.length === 1) return meaningful[0] as PredicateExpr;
  return { kind: "and", values: meaningful };
}

function orExpr(values: readonly PredicateExpr[]): PredicateExpr {
  const flattened = values.flatMap((value) => value.kind === "or" ? value.values : [value]);
  if (flattened.some((value) => value.kind === "bool" && value.value)) return { kind: "bool", value: true };
  const meaningful = flattened.filter((value) => value.kind !== "bool" || value.value);
  if (meaningful.length === 0) return { kind: "bool", value: false };
  if (meaningful.length === 1) return meaningful[0] as PredicateExpr;
  return { kind: "or", values: meaningful };
}

function notExpr(value: PredicateExpr): PredicateExpr {
  if (value.kind === "bool") return { kind: "bool", value: !value.value };
  if (value.kind === "not") return value.value;
  return { kind: "not", value };
}

function validatePermutation(axes: readonly number[], rank: number, path: string): readonly number[] {
  if (axes.length !== rank || axes.some((axis) => !Number.isSafeInteger(axis) || axis < 0 || axis >= rank) || new Set(axes).size !== rank) {
    invalid(path, `axes must be a permutation of [0, ${rank})`);
  }
  return [...axes];
}

function requireRank(values: readonly unknown[], state: NormalizationState, path: string): void {
  if (values.length > state.limits.maxRank) invalid(path, `rank ${values.length} exceeds limit ${state.limits.maxRank}`);
}

function requireTransformRank(values: readonly unknown[], rank: number, path: string): void {
  if (values.length !== rank) invalid(path, `transform rank ${values.length} must equal source rank ${rank}`);
}

function requireNonnegativeConstants(values: readonly DimExpr[], path: string): void {
  for (const [index, value] of values.entries()) {
    if (value.kind === "const" && wireIntegerToBigInt(value.value) < 0n) {
      invalid(`${path}[${index}]`, "extent must be non-negative");
    }
  }
}

function consume(state: NormalizationState, path: string, depth: number): void {
  state.nodes += 1;
  if (state.nodes > state.limits.maxNodes) resource(path, `normalization nodes exceed ${state.limits.maxNodes}`);
  if (depth > state.limits.maxDepth) resource(path, `normalization depth exceeds ${state.limits.maxDepth}`);
}

function isConstant(value: DimExpr, expected: bigint): boolean {
  return value.kind === "const" && wireIntegerToBigInt(value.value) === expected;
}

function isIndexConstant(value: IndexExpr, expected: bigint): boolean {
  return value.kind === "const" && wireIntegerToBigInt(value.value) === expected;
}

function sameDimExpr(left: DimExpr, right: DimExpr): boolean {
  return canonicalizeJson(left) === canonicalizeJson(right);
}

function invalid(path: string, message: string): never {
  throw new SemanticSchemaError({
    code: LAYOUT_DIAGNOSTIC_CODES.invalidLayoutExpr,
    stage: "verification",
    severity: "error",
    message,
    path,
  });
}

function resource(path: string, message: string): never {
  throw new SemanticSchemaError({
    code: LAYOUT_DIAGNOSTIC_CODES.resourceLimit,
    stage: "verification",
    severity: "error",
    message,
    path,
  });
}
