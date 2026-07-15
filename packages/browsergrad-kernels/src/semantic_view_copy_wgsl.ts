import type {
  IndexExpr,
  IndexMap,
  LayoutArtifactPayloadV1,
  PredicateExpr,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import type { PreparedViewCopySpecialization } from "@unlocalhosted/browsergrad-semantic-core/kernel";
import { wireIntegerToBigInt } from "@unlocalhosted/browsergrad-semantic-core/schema";

const I32_MIN = -2_147_483_648n;
const I32_MAX = 2_147_483_647n;

export interface EmittedSemanticViewCopyWgsl {
  readonly source: string;
  readonly sourceLocationRange: IntegerRange;
  readonly destinationLocationRange: IntegerRange;
}

export interface IntegerRange {
  readonly minimum: bigint;
  readonly maximum: bigint;
}

interface LoweredInteger extends IntegerRange {
  readonly code: string;
}

interface LoweringContext {
  readonly prepared: PreparedViewCopySpecialization;
  readonly path: string;
}

/** Lowers only the already verified and specialized initial i32 WGSL profile. */
export function emitSemanticViewCopyWgsl(
  layout: LayoutArtifactPayloadV1,
  prepared: PreparedViewCopySpecialization,
  workgroupSize: number,
): EmittedSemanticViewCopyWgsl {
  const sourceMap = requiredIndexMap(layout, prepared.source.indexMapId, "source");
  const destinationMap = requiredIndexMap(layout, prepared.destination.indexMapId, "destination");
  const sourceLocation = lowerIndex(sourceMap.location, {
    prepared,
    path: "$.source.indexMap.location",
  });
  const destinationLocation = lowerIndex(destinationMap.location, {
    prepared,
    path: "$.destination.indexMap.location",
  });
  const sourcePredicate = lowerPredicate(sourceMap.inBounds, {
    prepared,
    path: "$.source.indexMap.inBounds",
  });
  const destinationPredicate = lowerPredicate(destinationMap.inBounds, {
    prepared,
    path: "$.destination.indexMap.inBounds",
  });
  const sourceWord = lowerWordAddress(
    sourceLocation,
    sourceMap.locationUnit,
    prepared.source.viewByteOffset,
    "$.source",
  );
  const destinationWord = lowerWordAddress(
    destinationLocation,
    destinationMap.locationUnit,
    prepared.destination.viewByteOffset,
    "$.destination",
  );
  const elementCount = asU32(prepared.elementCount, "$.elementCount");
  const coordinates = emitCoordinates(prepared.logicalShape);
  const fillBits = prepared.operation.source.invalidSource.kind === "fill"
    ? `0x${prepared.operation.source.invalidSource.value.bits}u`
    : "0u";
  const copyBody = prepared.operation.source.invalidSource.kind === "fill"
    ? [
        `  var copied_bits: u32 = ${fillBits};`,
        `  if (${sourcePredicate}) {`,
        `    let source_word: u32 = ${sourceWord.code};`,
        "    copied_bits = source_words[source_word];",
        "  }",
      ]
    : [
        `  if (!(${sourcePredicate})) {`,
        "    return;",
        "  }",
        `  let source_word: u32 = ${sourceWord.code};`,
        "  let copied_bits: u32 = source_words[source_word];",
      ];
  const source = [
    "@group(0) @binding(0) var<storage, read> source_words: array<u32>;",
    "@group(0) @binding(1) var<storage, read_write> destination_words: array<u32>;",
    "",
    `@compute @workgroup_size(${workgroupSize}, 1, 1)`,
    "fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {",
    "  let linear_index: u32 = global_id.x;",
    `  if (linear_index >= ${elementCount}u) {`,
    "    return;",
    "  }",
    ...coordinates,
    `  if (!(${destinationPredicate})) {`,
    "    return;",
    "  }",
    `  let destination_word: u32 = ${destinationWord.code};`,
    ...copyBody,
    "  destination_words[destination_word] = copied_bits;",
    "}",
    "",
  ].join("\n");
  return Object.freeze({
    source,
    sourceLocationRange: Object.freeze({ minimum: sourceLocation.minimum, maximum: sourceLocation.maximum }),
    destinationLocationRange: Object.freeze({ minimum: destinationLocation.minimum, maximum: destinationLocation.maximum }),
  });
}

function lowerIndex(expression: IndexExpr, context: LoweringContext): LoweredInteger {
  switch (expression.kind) {
    case "const": {
      const value = wireIntegerToBigInt(expression.value);
      return checkedInteger(wgslI32(value), value, value, context.path);
    }
    case "dimension": {
      const binding = context.prepared.bindings[expression.symbolId];
      if (binding === undefined) unsupported(context.path, `missing specialized dimension ${expression.symbolId}`);
      const value = wireIntegerToBigInt(binding);
      return checkedInteger(wgslI32(value), value, value, context.path);
    }
    case "coordinate": {
      const extent = context.prepared.logicalShape[expression.axis];
      if (extent === undefined) unsupported(context.path, `coordinate axis ${expression.axis} is outside the prepared rank`);
      const maximum = extent === 0n ? 0n : extent - 1n;
      return checkedInteger(`coordinate_${expression.axis}`, 0n, maximum, context.path);
    }
    case "add": {
      const terms = expression.terms.map((term, index) => lowerIndex(term, child(context, `.terms[${index}]`)));
      const minimum = terms.reduce((sum, term) => sum + term.minimum, 0n);
      const maximum = terms.reduce((sum, term) => sum + term.maximum, 0n);
      return checkedInteger(`(${terms.map((term) => term.code).join(" + ")})`, minimum, maximum, context.path);
    }
    case "mul": {
      const lhs = lowerIndex(expression.lhs, child(context, ".lhs"));
      const rhs = lowerIndex(expression.rhs, child(context, ".rhs"));
      const products = [
        lhs.minimum * rhs.minimum,
        lhs.minimum * rhs.maximum,
        lhs.maximum * rhs.minimum,
        lhs.maximum * rhs.maximum,
      ];
      return checkedInteger(
        `(${lhs.code} * ${rhs.code})`,
        minimum(products),
        maximum(products),
        context.path,
      );
    }
    case "floorDiv":
    case "ceilDiv":
    case "mod":
    case "min":
    case "max":
      return unsupported(context.path, `${expression.kind} is outside browsergrad.webgpu.view-copy.i32@1`);
  }
}

function lowerPredicate(expression: PredicateExpr, context: LoweringContext): string {
  switch (expression.kind) {
    case "bool": return expression.value ? "true" : "false";
    case "equal": {
      const lhs = lowerIndex(expression.lhs, child(context, ".lhs"));
      const rhs = lowerIndex(expression.rhs, child(context, ".rhs"));
      return `(${lhs.code} == ${rhs.code})`;
    }
    case "lessEqual": {
      const lhs = lowerIndex(expression.lhs, child(context, ".lhs"));
      const rhs = lowerIndex(expression.rhs, child(context, ".rhs"));
      return `(${lhs.code} <= ${rhs.code})`;
    }
    case "and":
    case "or": {
      const operator = expression.kind === "and" ? " && " : " || ";
      const values = expression.values.map((value, index) => lowerPredicate(value, child(context, `.values[${index}]`)));
      return `(${values.join(operator)})`;
    }
    case "not": return `!(${lowerPredicate(expression.value, child(context, ".value"))})`;
  }
}

function lowerWordAddress(
  location: LoweredInteger,
  unit: "element" | "byte",
  viewByteOffset: bigint,
  path: string,
): LoweredInteger {
  if (viewByteOffset % 4n !== 0n) unsupported(`${path}.viewByteOffset`, "f32 view byte offset must be word aligned");
  const byteScale = unit === "element" ? 4n : 1n;
  const scaledMinimum = location.minimum * byteScale;
  const scaledMaximum = location.maximum * byteScale;
  checkedRange(scaledMinimum, scaledMaximum, `${path}.scaledLocation`);
  const byteMinimum = viewByteOffset + scaledMinimum;
  const byteMaximum = viewByteOffset + scaledMaximum;
  checkedRange(byteMinimum, byteMaximum, `${path}.byteAddress`);
  const scaleCode = unit === "element" ? `(${location.code} * 4i)` : location.code;
  const byteCode = `(${wgslI32(viewByteOffset)} + ${scaleCode})`;
  return {
    code: `u32(${byteCode} / 4i)`,
    minimum: divideExactly(byteMinimum, 4n, `${path}.wordAddress`),
    maximum: divideExactly(byteMaximum, 4n, `${path}.wordAddress`),
  };
}

function emitCoordinates(shape: readonly bigint[]): readonly string[] {
  if (shape.some((extent) => extent === 0n)) {
    return shape.map((_, axis) => `  let coordinate_${axis}: i32 = 0i;`);
  }
  if (shape.length === 2) {
    const inner = asU32(shape[1] as bigint, "$.shape[1]");
    return [
      `  let coordinate_0: i32 = i32(linear_index / ${inner}u);`,
      `  let coordinate_1: i32 = i32(linear_index % ${inner}u);`,
    ];
  }
  if (shape.length === 3) {
    const middle = asU32(shape[1] as bigint, "$.shape[1]");
    const inner = asU32(shape[2] as bigint, "$.shape[2]");
    const plane = asU32(BigInt(middle) * BigInt(inner), "$.shape[1:3]");
    return [
      `  let coordinate_0: i32 = i32(linear_index / ${plane}u);`,
      `  let plane_remainder: u32 = linear_index % ${plane}u;`,
      `  let coordinate_1: i32 = i32(plane_remainder / ${inner}u);`,
      `  let coordinate_2: i32 = i32(plane_remainder % ${inner}u);`,
    ];
  }
  return unsupported("$.shape", "WGSL view-copy supports rank 2 or 3 only");
}

function requiredIndexMap(layout: LayoutArtifactPayloadV1, indexMapId: string, role: string): IndexMap {
  const indexMap = layout.indexMaps.find((entry) => entry.indexMapId === indexMapId);
  if (indexMap === undefined) throw new Error(`internal: prepared ${role} index map disappeared`);
  return indexMap;
}

function child(context: LoweringContext, suffix: string): LoweringContext {
  return { ...context, path: `${context.path}${suffix}` };
}

function checkedInteger(code: string, minimumValue: bigint, maximumValue: bigint, path: string): LoweredInteger {
  checkedRange(minimumValue, maximumValue, path);
  return { code, minimum: minimumValue, maximum: maximumValue };
}

function checkedRange(minimumValue: bigint, maximumValue: bigint, path: string): void {
  if (minimumValue < I32_MIN || maximumValue > I32_MAX) {
    unsupported(path, `resolved integer range [${minimumValue}, ${maximumValue}] exceeds exact WGSL i32 arithmetic`);
  }
}

function wgslI32(value: bigint): string {
  checkedRange(value, value, "$.literal");
  if (value === I32_MIN) return "bitcast<i32>(0x80000000u)";
  return `${value}i`;
}

function asU32(value: bigint, path: string): number {
  if (value < 0n || value > 0xffff_ffffn) unsupported(path, `resolved value ${value} exceeds WGSL u32`);
  return Number(value);
}

function divideExactly(value: bigint, divisor: bigint, path: string): bigint {
  if (value % divisor !== 0n) unsupported(path, `resolved byte address ${value} is not word aligned`);
  return value / divisor;
}

function minimum(values: readonly bigint[]): bigint {
  return values.reduce((result, value) => value < result ? value : result);
}

function maximum(values: readonly bigint[]): bigint {
  return values.reduce((result, value) => value > result ? value : result);
}

function unsupported(path: string, message: string): never {
  throw new SemanticViewCopyWgslLoweringError(path, message);
}

export class SemanticViewCopyWgslLoweringError extends Error {
  readonly code = "BG-WEBGPU-VIEW-COPY-UNSUPPORTED-PROFILE" as const;

  constructor(readonly path: string, message: string) {
    super(message);
    this.name = "SemanticViewCopyWgslLoweringError";
  }
}
