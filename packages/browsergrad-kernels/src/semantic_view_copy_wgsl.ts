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

export const SEMANTIC_VIEW_COPY_DYNAMIC_PREFIX_UNIFORM =
  "bg_dynamic_prefix" as const;
export const SEMANTIC_VIEW_COPY_DYNAMIC_PREFIX_BINDING = 2;
export const SEMANTIC_VIEW_COPY_DYNAMIC_REGION_UNIFORM =
  "bg_dynamic_region" as const;
export const SEMANTIC_VIEW_COPY_DYNAMIC_REGION_BINDING = 2;

export type SemanticViewCopyLaunchMode =
  | "static"
  | "runtime-linear-prefix"
  | "runtime-rectangular-prefix";

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

/** Lowers only an already verified and specialized exact 32-bit word profile. */
export function emitSemanticViewCopyWgsl(
  layout: LayoutArtifactPayloadV1,
  prepared: PreparedViewCopySpecialization,
  workgroupSize: number,
  launchMode: SemanticViewCopyLaunchMode = "static",
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
  const launch = emitLaunchPrelude(
    prepared.logicalShape,
    elementCount,
    launchMode,
  );
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
    ...launch.declarations,
    "",
    `@compute @workgroup_size(${workgroupSize}, 1, 1)`,
    "fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {",
    ...launch.body,
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

function emitLaunchPrelude(
  shape: readonly bigint[],
  elementCount: number,
  mode: SemanticViewCopyLaunchMode,
): Readonly<{
  readonly declarations: readonly string[];
  readonly body: readonly string[];
}> {
  if (mode === "runtime-rectangular-prefix") {
    if (
      shape.length !== 2 &&
      shape.length !== 3 &&
      shape.length !== 4
    ) {
      return unsupported(
        "$.shape",
        "rectangular dynamic WGSL launch supports semantic ranks 2 through 4 only",
      );
    }
    const staticExtents = shape.map((extent, axis) =>
      asU32(extent, `$.shape[${axis}]`));
    const declarations = Object.freeze([
      "struct BrowserGradDynamicRegion {",
      "  extent_0: u32,",
      "  extent_1: u32,",
      "  extent_2: u32,",
      "  extent_3: u32,",
      "}",
      `@group(0) @binding(${SEMANTIC_VIEW_COPY_DYNAMIC_REGION_BINDING}) var<uniform> ${SEMANTIC_VIEW_COPY_DYNAMIC_REGION_UNIFORM}: BrowserGradDynamicRegion;`,
    ]);
    if (shape.length === 4) {
      const leadingStaticProduct = asU32(
        shape[0]! * shape[1]!,
        "$.shape[0:2]",
      );
      return Object.freeze({
        declarations,
        body: Object.freeze([
          `  if (global_id.x >= ${staticExtents[3]}u || global_id.x >= ${SEMANTIC_VIEW_COPY_DYNAMIC_REGION_UNIFORM}.extent_3 || global_id.y >= ${staticExtents[2]}u || global_id.y >= ${SEMANTIC_VIEW_COPY_DYNAMIC_REGION_UNIFORM}.extent_2 || global_id.z >= ${leadingStaticProduct}u || global_id.z >= (${SEMANTIC_VIEW_COPY_DYNAMIC_REGION_UNIFORM}.extent_0 * ${SEMANTIC_VIEW_COPY_DYNAMIC_REGION_UNIFORM}.extent_1)) {`,
          "    return;",
          "  }",
          `  let coordinate_0: i32 = i32(global_id.z / ${SEMANTIC_VIEW_COPY_DYNAMIC_REGION_UNIFORM}.extent_1);`,
          `  let coordinate_1: i32 = i32(global_id.z % ${SEMANTIC_VIEW_COPY_DYNAMIC_REGION_UNIFORM}.extent_1);`,
          "  let coordinate_2: i32 = i32(global_id.y);",
          "  let coordinate_3: i32 = i32(global_id.x);",
        ]),
      });
    }
    const globalAxes = shape.length === 2
      ? ["y", "x"] as const
      : ["z", "y", "x"] as const;
    const bounds = globalAxes.flatMap((globalAxis, axis) => [
      `global_id.${globalAxis} >= ${staticExtents[axis]}u`,
      `global_id.${globalAxis} >= ${SEMANTIC_VIEW_COPY_DYNAMIC_REGION_UNIFORM}.extent_${axis}`,
    ]);
    return Object.freeze({
      declarations,
      body: Object.freeze([
        `  if (${bounds.join(" || ")}) {`,
        "    return;",
        "  }",
        ...globalAxes.map((globalAxis, axis) =>
          `  let coordinate_${axis}: i32 = i32(global_id.${globalAxis});`),
      ]),
    });
  }
  const declarations = mode === "runtime-linear-prefix"
    ? [
        "struct BrowserGradDynamicPrefix {",
        "  element_count: u32,",
        "}",
        `@group(0) @binding(${SEMANTIC_VIEW_COPY_DYNAMIC_PREFIX_BINDING}) var<uniform> ${SEMANTIC_VIEW_COPY_DYNAMIC_PREFIX_UNIFORM}: BrowserGradDynamicPrefix;`,
      ]
    : [];
  const elementCountGuard = mode === "runtime-linear-prefix"
    ? `  if (linear_index >= ${elementCount}u || linear_index >= ${SEMANTIC_VIEW_COPY_DYNAMIC_PREFIX_UNIFORM}.element_count) {`
    : `  if (linear_index >= ${elementCount}u) {`;
  return Object.freeze({
    declarations: Object.freeze(declarations),
    body: Object.freeze([
      "  let linear_index: u32 = global_id.x;",
      elementCountGuard,
      "    return;",
      "  }",
      ...emitCoordinates(shape),
    ]),
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
      return unsupported(
        context.path,
        `${expression.kind} is outside browsergrad.webgpu.view-copy.word32@2`,
      );
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
  if (viewByteOffset % 4n !== 0n) {
    unsupported(
      `${path}.viewByteOffset`,
      "32-bit view byte offset must be word aligned",
    );
  }
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
  if (shape.length === 1) {
    return ["  let coordinate_0: i32 = i32(linear_index);"];
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
  if (shape.length === 4) {
    const second = asU32(shape[1] as bigint, "$.shape[1]");
    const third = asU32(shape[2] as bigint, "$.shape[2]");
    const fourth = asU32(shape[3] as bigint, "$.shape[3]");
    const innerPlane = asU32(
      BigInt(third) * BigInt(fourth),
      "$.shape[2:4]",
    );
    const outerPlane = asU32(
      BigInt(second) * BigInt(innerPlane),
      "$.shape[1:4]",
    );
    return [
      `  let coordinate_0: i32 = i32(linear_index / ${outerPlane}u);`,
      `  let outer_remainder: u32 = linear_index % ${outerPlane}u;`,
      `  let coordinate_1: i32 = i32(outer_remainder / ${innerPlane}u);`,
      `  let inner_remainder: u32 = outer_remainder % ${innerPlane}u;`,
      `  let coordinate_2: i32 = i32(inner_remainder / ${fourth}u);`,
      `  let coordinate_3: i32 = i32(inner_remainder % ${fourth}u);`,
    ];
  }
  if (shape.length === 5) {
    const second = asU32(shape[1] as bigint, "$.shape[1]");
    const third = asU32(shape[2] as bigint, "$.shape[2]");
    const fourth = asU32(shape[3] as bigint, "$.shape[3]");
    const fifth = asU32(shape[4] as bigint, "$.shape[4]");
    const stride2 = asU32(
      BigInt(fourth) * BigInt(fifth),
      "$.shape[3:5]",
    );
    const stride1 = asU32(
      BigInt(third) * BigInt(stride2),
      "$.shape[2:5]",
    );
    const stride0 = asU32(
      BigInt(second) * BigInt(stride1),
      "$.shape[1:5]",
    );
    return [
      `  let coordinate_0: i32 = i32(linear_index / ${stride0}u);`,
      `  let rank5_remainder_0: u32 = linear_index % ${stride0}u;`,
      `  let coordinate_1: i32 = i32(rank5_remainder_0 / ${stride1}u);`,
      `  let rank5_remainder_1: u32 = rank5_remainder_0 % ${stride1}u;`,
      `  let coordinate_2: i32 = i32(rank5_remainder_1 / ${stride2}u);`,
      `  let rank5_remainder_2: u32 = rank5_remainder_1 % ${stride2}u;`,
      `  let coordinate_3: i32 = i32(rank5_remainder_2 / ${fifth}u);`,
      `  let coordinate_4: i32 = i32(rank5_remainder_2 % ${fifth}u);`,
    ];
  }
  return unsupported("$.shape", "WGSL view-copy supports ranks in [1, 5] only");
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
