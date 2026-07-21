import type { PreparedLogicalGemmTileSpecialization } from "@unlocalhosted/browsergrad-semantic-core/kernel";
import type { PreparedLogicalGemmTileSchedule } from "@unlocalhosted/browsergrad-semantic-core/schedule";

const U32_MAX = 0xffff_ffffn;

export interface EmittedSemanticGemmWgsl {
  readonly source: string;
  readonly workgroupSize: readonly [number, number, number];
  readonly workgroupStorageBytes: bigint;
}

export type SemanticGemmWgslLoweringErrorCode =
  | "BG-WEBGPU-GEMM-UNSUPPORTED-PROFILE"
  | "BG-WEBGPU-GEMM-RESOURCE-LIMIT";

export class SemanticGemmWgslLoweringError extends Error {
  constructor(
    readonly code: SemanticGemmWgslLoweringErrorCode,
    readonly path: string,
    message: string,
  ) {
    super(message);
    this.name = "SemanticGemmWgslLoweringError";
  }
}

/**
 * Lowers the verified dense-rank-2 f32 GEMM profile and its canonical scalar
 * cooperative schedule. This function selects no logical or numerical facts;
 * it only realizes the already verified physical mapping.
 */
export function emitSemanticGemmWgsl(
  prepared: PreparedLogicalGemmTileSpecialization,
  scheduled: PreparedLogicalGemmTileSchedule,
): EmittedSemanticGemmWgsl {
  if (scheduled.logical !== prepared) {
    unsupported(
      "$.schedule.logical",
      "WGSL lowering requires the exact logical specialization authorized by schedule preparation",
    );
  }
  const schedule = scheduled.schedule;
  if (
    schedule.participation.workgroup !== "all-invocations"
    || schedule.participation.earlyExit !== "forbidden"
    || schedule.uniformity.barrierControl !== "workgroup-uniform"
    || schedule.masks.lhsLoad !== "zero-fill"
    || schedule.masks.rhsLoad !== "zero-fill"
    || schedule.masks.destinationStore !== "suppress"
  ) {
    unsupported("$.schedule", "prepared schedule is outside the uniform scalar cooperative WGSL profile");
  }
  const m = asU32(prepared.m, "$.semantic.m");
  const n = asU32(prepared.n, "$.semantic.n");
  const k = asU32(prepared.k, "$.semantic.k");
  const lhsOffset = wordOffset(prepared.lhs.viewByteOffset, "$.semantic.lhs.viewByteOffset");
  const rhsOffset = wordOffset(prepared.rhs.viewByteOffset, "$.semantic.rhs.viewByteOffset");
  const destinationOffset = wordOffset(
    prepared.destination.viewByteOffset,
    "$.semantic.destination.viewByteOffset",
  );
  requireLastWord(lhsOffset, prepared.m * prepared.k, "$.semantic.lhs");
  requireLastWord(rhsOffset, prepared.k * prepared.n, "$.semantic.rhs");
  requireLastWord(destinationOffset, prepared.m * prepared.n, "$.semantic.destination");

  const tileM = asU32(scheduled.physicalM, "$.schedule.physicalTile.m");
  const tileN = asU32(scheduled.physicalN, "$.schedule.physicalTile.n");
  const tileK = asU32(scheduled.physicalK, "$.schedule.physicalTile.k");
  const workgroupX = asU32(scheduled.workgroupSizeX, "$.schedule.workgroup.size.x");
  const workgroupY = asU32(scheduled.workgroupSizeY, "$.schedule.workgroup.size.y");
  const workgroupZ = asU32(scheduled.workgroupSizeZ, "$.schedule.workgroup.size.z");
  if (workgroupX !== tileN || workgroupY !== tileM || workgroupZ !== 1) {
    unsupported("$.schedule.workgroup.size", "one-output-element lowering requires N by M by 1 workgroups");
  }
  const invocationCount = checkedU32Positive(
    scheduled.workgroupInvocations,
    "$.schedule.workgroup.size",
  );
  const lhsTileElements = checkedU32Positive(
    scheduled.lhsStagingElements,
    "$.schedule.staging.lhs",
  );
  const rhsTileElements = checkedU32Positive(
    scheduled.rhsStagingElements,
    "$.schedule.staging.rhs",
  );
  const workgroupStorageBytes = scheduled.aggregateStagingElements * 4n;
  requirePaddedInvocationRange(prepared.m, BigInt(tileM), "$.semantic.m");
  requirePaddedInvocationRange(prepared.n, BigInt(tileN), "$.semantic.n");
  const kTileCount = (prepared.k / BigInt(tileK))
    + (prepared.k % BigInt(tileK) === 0n ? 0n : 1n);

  const source = [
    "@group(0) @binding(0) var<storage, read> lhs_values: array<f32>;",
    "@group(0) @binding(1) var<storage, read> rhs_values: array<f32>;",
    "@group(0) @binding(2) var<storage, read_write> destination_values: array<f32>;",
    "",
    `var<workgroup> lhs_tile: array<f32, ${lhsTileElements}>;`,
    `var<workgroup> rhs_tile: array<f32, ${rhsTileElements}>;`,
    "",
    `@compute @workgroup_size(${workgroupX}, ${workgroupY}, 1)`,
    "fn main(",
    "  @builtin(local_invocation_id) local_id: vec3<u32>,",
    "  @builtin(workgroup_id) workgroup_id: vec3<u32>,",
    ") {",
    `  let output_row: u32 = workgroup_id.y * ${tileM}u + local_id.y;`,
    `  let output_column: u32 = workgroup_id.x * ${tileN}u + local_id.x;`,
    `  let linear_lane: u32 = local_id.y * ${tileN}u + local_id.x;`,
    "  var accumulator: f32 = 0.0;",
    "",
    `  for (var tile_index: u32 = 0u; tile_index < ${kTileCount}u; tile_index += 1u) {`,
    `    let tile_k_base: u32 = tile_index * ${tileK}u;`,
    `    for (var load_index: u32 = linear_lane; load_index < ${lhsTileElements}u; load_index += ${invocationCount}u) {`,
    `      let tile_row: u32 = load_index / ${tileK}u;`,
    `      let tile_inner: u32 = load_index % ${tileK}u;`,
    `      let global_row: u32 = workgroup_id.y * ${tileM}u + tile_row;`,
    "      var value: f32 = 0.0;",
    `      if (global_row < ${m}u && tile_inner < (${k}u - tile_k_base)) {`,
    "        let global_inner: u32 = tile_k_base + tile_inner;",
    `        value = lhs_values[${lhsOffset}u + global_row * ${k}u + global_inner];`,
    "      }",
    "      lhs_tile[load_index] = value;",
    "    }",
    "",
    `    for (var load_index: u32 = linear_lane; load_index < ${rhsTileElements}u; load_index += ${invocationCount}u) {`,
    `      let tile_inner: u32 = load_index / ${tileN}u;`,
    `      let tile_column: u32 = load_index % ${tileN}u;`,
    `      let global_column: u32 = workgroup_id.x * ${tileN}u + tile_column;`,
    "      var value: f32 = 0.0;",
    `      if (tile_inner < (${k}u - tile_k_base) && global_column < ${n}u) {`,
    "        let global_inner: u32 = tile_k_base + tile_inner;",
    `        value = rhs_values[${rhsOffset}u + global_inner * ${n}u + global_column];`,
    "      }",
    "      rhs_tile[load_index] = value;",
    "    }",
    "",
    "    workgroupBarrier();",
    `    for (var tile_inner: u32 = 0u; tile_inner < ${tileK}u; tile_inner += 1u) {`,
    `      let lhs_value: f32 = lhs_tile[local_id.y * ${tileK}u + tile_inner];`,
    `      let rhs_value: f32 = rhs_tile[tile_inner * ${tileN}u + local_id.x];`,
    "      accumulator = accumulator + (lhs_value * rhs_value);",
    "    }",
    "    workgroupBarrier();",
    "  }",
    "",
    `  if (output_row < ${m}u && output_column < ${n}u) {`,
    `    destination_values[${destinationOffset}u + output_row * ${n}u + output_column] = accumulator;`,
    "  }",
    "}",
    "",
  ].join("\n");

  return Object.freeze({
    source,
    workgroupSize: Object.freeze([workgroupX, workgroupY, 1] as const),
    workgroupStorageBytes,
  });
}

function wordOffset(byteOffset: bigint, path: string): number {
  if (byteOffset % 4n !== 0n) unsupported(path, "dense f32 byte offset must be four-byte aligned");
  return asU32(byteOffset / 4n, path);
}

function requireLastWord(offset: number, elementCount: bigint, path: string): void {
  if (elementCount === 0n) return;
  const last = BigInt(offset) + elementCount - 1n;
  if (last > U32_MAX) unsupported(path, "dense f32 address exceeds the initial u32 WGSL profile");
}

function requirePaddedInvocationRange(
  extent: bigint,
  physicalTile: bigint,
  path: string,
): void {
  const padded = ((extent + physicalTile - 1n) / physicalTile) * physicalTile;
  if (padded - 1n > U32_MAX) {
    unsupported(path, "padded workgroup mapping exceeds the initial u32 WGSL invocation range");
  }
}

function asU32(value: bigint, path: string): number {
  if (value < 0n || value > U32_MAX) unsupported(path, "value is outside the initial u32 WGSL profile");
  return Number(value);
}

function checkedU32Positive(value: bigint, path: string): bigint {
  if (value === 0n || value > U32_MAX) {
    unsupported(path, "prepared schedule value is outside the initial u32 WGSL profile");
  }
  return value;
}

function unsupported(path: string, message: string): never {
  throw new SemanticGemmWgslLoweringError(
    "BG-WEBGPU-GEMM-UNSUPPORTED-PROFILE",
    path,
    message,
  );
}
