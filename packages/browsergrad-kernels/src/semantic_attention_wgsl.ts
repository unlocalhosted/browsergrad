import type { PreparedAttentionForwardSpecialization } from "@unlocalhosted/browsergrad-semantic-core/kernel";
import type { PreparedAttentionOnlineKvTileSchedule } from "@unlocalhosted/browsergrad-semantic-core/schedule";

const U32_MAX = 0xffff_ffffn;

export interface EmittedSemanticAttentionWgsl {
  readonly source: string;
  readonly workgroupSize: readonly [number, number, number];
  readonly workgroupStorageBytes: bigint;
}

export type SemanticAttentionWgslLoweringErrorCode =
  | "BG-WEBGPU-ATTENTION-UNSUPPORTED-PROFILE"
  | "BG-WEBGPU-ATTENTION-RESOURCE-LIMIT";

export class SemanticAttentionWgslLoweringError extends Error {
  constructor(
    readonly code: SemanticAttentionWgslLoweringErrorCode,
    readonly path: string,
    message: string,
  ) {
    super(message);
    this.name = "SemanticAttentionWgslLoweringError";
  }
}

/**
 * Lowers the exact dense rank-4 f32 attention meaning and its independently
 * verified scalar online-K/V-tile schedule. Logical mask, scale, reduction,
 * and effect facts are consumed rather than reconstructed from caller input.
 */
export function emitSemanticAttentionWgsl(
  semantic: PreparedAttentionForwardSpecialization,
  scheduled: PreparedAttentionOnlineKvTileSchedule,
): EmittedSemanticAttentionWgsl {
  if (scheduled.logical !== semantic) {
    unsupported(
      "$.schedule.logical",
      "WGSL lowering requires the exact logical specialization authorized by schedule preparation",
    );
  }
  requireScheduleProfile(scheduled);

  const batch = asU32(semantic.batch, "$.semantic.batch");
  const heads = asU32(semantic.heads, "$.semantic.heads");
  const queryLength = asU32(semantic.queryLength, "$.semantic.queryLength");
  const keyLength = asU32(semantic.keyLength, "$.semantic.keyLength");
  const queryDepth = asU32(semantic.queryDepth, "$.semantic.queryDepth");
  const valueDepth = asU32(semantic.valueDepth, "$.semantic.valueDepth");
  const queryOffset = wordOffset(semantic.query.viewByteOffset, "$.semantic.query.viewByteOffset");
  const keyOffset = wordOffset(semantic.key.viewByteOffset, "$.semantic.key.viewByteOffset");
  const valueOffset = wordOffset(semantic.value.viewByteOffset, "$.semantic.value.viewByteOffset");
  const destinationOffset = wordOffset(
    semantic.destination.viewByteOffset,
    "$.semantic.destination.viewByteOffset",
  );
  requireLastWord(queryOffset, semantic.queryElements, "$.semantic.query");
  requireLastWord(keyOffset, semantic.keyElements, "$.semantic.key");
  requireLastWord(valueOffset, semantic.valueElements, "$.semantic.value");
  requireLastWord(destinationOffset, semantic.outputElements, "$.semantic.destination");

  const queryRows = asU32(scheduled.queryRows, "$.schedule.physicalTile.queryRows");
  const keyRows = asU32(scheduled.keyRows, "$.schedule.physicalTile.keyRows");
  const workgroupX = asU32(scheduled.workgroupSizeX, "$.schedule.workgroup.size.x");
  const workgroupY = asU32(scheduled.workgroupSizeY, "$.schedule.workgroup.size.y");
  const workgroupZ = asU32(scheduled.workgroupSizeZ, "$.schedule.workgroup.size.z");
  if (workgroupX !== queryRows || workgroupY !== 1 || workgroupZ !== 1) {
    unsupported(
      "$.schedule.workgroup.size",
      "one-query-row lowering requires queryRows by 1 by 1 workgroups",
    );
  }
  const invocationCount = checkedU32Positive(
    scheduled.workgroupInvocations,
    "$.schedule.workgroup.size",
  );
  const keyStagingElements = checkedU32Positive(
    scheduled.keyStagingElements,
    "$.schedule.staging.key",
  );
  const valueStagingElements = checkedU32Positive(
    scheduled.valueStagingElements,
    "$.schedule.staging.value",
  );
  const keyTiles = checkedU32Positive(scheduled.keyTiles, "$.schedule.keyTiles");
  const workgroupStorageBytes = scheduled.aggregateStagingBytes;
  const causalPredicate = semantic.operation.mask.kind === "causal"
    ? " && global_key <= query_index"
    : "";
  const scaleBits = semantic.operation.scale.value.bits;
  if (!/^[0-9a-f]{8}$/u.test(scaleBits)) {
    unsupported("$.semantic.scale", "attention scale must be one canonical f32 bit pattern");
  }

  const source = [
    "@group(0) @binding(0) var<storage, read> query_values: array<f32>;",
    "@group(0) @binding(1) var<storage, read> key_values: array<f32>;",
    "@group(0) @binding(2) var<storage, read> value_values: array<f32>;",
    "@group(0) @binding(3) var<storage, read_write> destination_values: array<f32>;",
    "",
    `var<workgroup> key_tile: array<f32, ${keyStagingElements}>;`,
    `var<workgroup> value_tile: array<f32, ${valueStagingElements}>;`,
    "",
    `@compute @workgroup_size(${workgroupX}, 1, 1)`,
    "fn main(",
    "  @builtin(local_invocation_id) local_id: vec3<u32>,",
    "  @builtin(workgroup_id) workgroup_id: vec3<u32>,",
    ") {",
    `  let query_index: u32 = workgroup_id.x * ${queryRows}u + local_id.x;`,
    "  let head: u32 = workgroup_id.y;",
    "  let batch_index: u32 = workgroup_id.z;",
    `  let active_query: bool = query_index < ${queryLength}u && head < ${heads}u && batch_index < ${batch}u;`,
    `  let batch_head: u32 = batch_index * ${heads}u + head;`,
    "  let linear_lane: u32 = local_id.x;",
    `  let scale: f32 = bitcast<f32>(0x${scaleBits}u);`,
    `  var query_private: array<f32, ${queryDepth}>;`,
    `  var output_private: array<f32, ${valueDepth}>;`,
    `  for (var depth: u32 = 0u; depth < ${queryDepth}u; depth += 1u) {`,
    "    query_private[depth] = 0.0;",
    "    if (active_query) {",
    `      let query_address: u32 = ${queryOffset}u + ((batch_head * ${queryLength}u + query_index) * ${queryDepth}u) + depth;`,
    "      query_private[depth] = query_values[query_address];",
    "    }",
    "  }",
    `  for (var value_index: u32 = 0u; value_index < ${valueDepth}u; value_index += 1u) {`,
    "    output_private[value_index] = 0.0;",
    "  }",
    "  var has_state: bool = false;",
    "  var running_maximum: f32 = 0.0;",
    "  var denominator: f32 = 0.0;",
    "",
    `  for (var tile_index: u32 = 0u; tile_index < ${keyTiles}u; tile_index += 1u) {`,
    `    let tile_key_base: u32 = tile_index * ${keyRows}u;`,
    `    for (var load_index: u32 = linear_lane; load_index < ${keyStagingElements}u; load_index += ${invocationCount}u) {`,
    `      let tile_key: u32 = load_index / ${queryDepth}u;`,
    `      let depth: u32 = load_index % ${queryDepth}u;`,
    "      let global_key: u32 = tile_key_base + tile_key;",
    "      var loaded: f32 = 0.0;",
    `      if (global_key < ${keyLength}u) {`,
    `        let key_address: u32 = ${keyOffset}u + ((batch_head * ${keyLength}u + global_key) * ${queryDepth}u) + depth;`,
    "        loaded = key_values[key_address];",
    "      }",
    "      key_tile[load_index] = loaded;",
    "    }",
    `    for (var load_index: u32 = linear_lane; load_index < ${valueStagingElements}u; load_index += ${invocationCount}u) {`,
    `      let tile_key: u32 = load_index / ${valueDepth}u;`,
    `      let value_index: u32 = load_index % ${valueDepth}u;`,
    "      let global_key: u32 = tile_key_base + tile_key;",
    "      var loaded: f32 = 0.0;",
    `      if (global_key < ${keyLength}u) {`,
    `        let value_address: u32 = ${valueOffset}u + ((batch_head * ${keyLength}u + global_key) * ${valueDepth}u) + value_index;`,
    "        loaded = value_values[value_address];",
    "      }",
    "      value_tile[load_index] = loaded;",
    "    }",
    "",
    "    workgroupBarrier();",
    "    if (active_query) {",
    "      var tile_has_valid: bool = false;",
    "      var tile_maximum: f32 = 0.0;",
    `      for (var tile_key: u32 = 0u; tile_key < ${keyRows}u; tile_key += 1u) {`,
    "        let global_key: u32 = tile_key_base + tile_key;",
    `        let valid_key: bool = global_key < ${keyLength}u${causalPredicate};`,
    "        if (valid_key) {",
    "          var score: f32 = 0.0;",
    `          for (var depth: u32 = 0u; depth < ${queryDepth}u; depth += 1u) {`,
    `            score += query_private[depth] * key_tile[tile_key * ${queryDepth}u + depth];`,
    "          }",
    "          score *= scale;",
    "          if (!tile_has_valid || score > tile_maximum) {",
    "            tile_maximum = score;",
    "            tile_has_valid = true;",
    "          }",
    "        }",
    "      }",
    "      if (tile_has_valid) {",
    "        var next_maximum: f32 = tile_maximum;",
    "        var prior_rescale: f32 = 0.0;",
    "        if (has_state) {",
    "          next_maximum = max(running_maximum, tile_maximum);",
    "          prior_rescale = exp(running_maximum - next_maximum);",
    "        }",
    "        denominator *= prior_rescale;",
    `        for (var value_index: u32 = 0u; value_index < ${valueDepth}u; value_index += 1u) {`,
    "          output_private[value_index] *= prior_rescale;",
    "        }",
    `        for (var tile_key: u32 = 0u; tile_key < ${keyRows}u; tile_key += 1u) {`,
    "          let global_key: u32 = tile_key_base + tile_key;",
    `          let valid_key: bool = global_key < ${keyLength}u${causalPredicate};`,
    "          if (valid_key) {",
    "            var score: f32 = 0.0;",
    `            for (var depth: u32 = 0u; depth < ${queryDepth}u; depth += 1u) {`,
    `              score += query_private[depth] * key_tile[tile_key * ${queryDepth}u + depth];`,
    "            }",
    "            score *= scale;",
    "            let weight: f32 = exp(score - next_maximum);",
    "            denominator += weight;",
    `            for (var value_index: u32 = 0u; value_index < ${valueDepth}u; value_index += 1u) {`,
    `              output_private[value_index] += weight * value_tile[tile_key * ${valueDepth}u + value_index];`,
    "            }",
    "          }",
    "        }",
    "        running_maximum = next_maximum;",
    "        has_state = true;",
    "      }",
    "    }",
    "    workgroupBarrier();",
    "  }",
    "",
    "  if (active_query && has_state && denominator > 0.0) {",
    `    for (var value_index: u32 = 0u; value_index < ${valueDepth}u; value_index += 1u) {`,
    `      let destination_address: u32 = ${destinationOffset}u + ((batch_head * ${queryLength}u + query_index) * ${valueDepth}u) + value_index;`,
    "      destination_values[destination_address] = output_private[value_index] / denominator;",
    "    }",
    "  }",
    "}",
    "",
  ].join("\n");

  return Object.freeze({
    source,
    workgroupSize: Object.freeze([workgroupX, 1, 1] as const),
    workgroupStorageBytes,
  });
}

function requireScheduleProfile(scheduled: PreparedAttentionOnlineKvTileSchedule): void {
  const schedule = scheduled.schedule;
  if (
    schedule.participation.workgroup !== "all-invocations"
    || schedule.participation.boundaryQueryLanes !== "participate"
    || schedule.participation.earlyExit !== "forbidden"
    || schedule.uniformity.barrierControl !== "workgroup-uniform"
    || schedule.uniformity.activeMaskScope !== "memory-effects-and-online-state-only"
    || schedule.staging.space !== "workgroup"
    || schedule.staging.key !== "cooperative"
    || schedule.staging.value !== "cooperative"
    || schedule.staging.buffering !== "single"
    || schedule.masks.keyLoad !== "zero-fill"
    || schedule.masks.valueLoad !== "zero-fill"
    || schedule.masks.invalidKeyScore !== "exclude-before-online-state-update"
    || schedule.masks.logicalMask !== "exclude-before-online-state-update"
    || schedule.masks.destinationStore !== "suppress"
    || schedule.vectorization.keyLoad !== "1"
    || schedule.vectorization.valueLoad !== "1"
    || schedule.vectorization.destinationStore !== "1"
  ) {
    unsupported(
      "$.schedule",
      "prepared schedule is outside the scalar cooperative online K/V-tile WGSL profile",
    );
  }
}

function wordOffset(byteOffset: bigint, path: string): number {
  if (byteOffset % 4n !== 0n) unsupported(path, "dense f32 byte offset must be four-byte aligned");
  return asU32(byteOffset / 4n, path);
}

function requireLastWord(offset: number, elementCount: bigint, path: string): void {
  if (elementCount === 0n) return;
  if (BigInt(offset) + elementCount - 1n > U32_MAX) {
    unsupported(path, "dense f32 address exceeds the initial u32 WGSL profile");
  }
}

function asU32(value: bigint, path: string): number {
  if (value < 0n || value > U32_MAX) {
    unsupported(path, "value is outside the initial u32 WGSL profile");
  }
  return Number(value);
}

function checkedU32Positive(value: bigint, path: string): bigint {
  if (value === 0n || value > U32_MAX) {
    unsupported(path, "prepared schedule value is outside the initial u32 WGSL profile");
  }
  return value;
}

function unsupported(path: string, message: string): never {
  throw new SemanticAttentionWgslLoweringError(
    "BG-WEBGPU-ATTENTION-UNSUPPORTED-PROFILE",
    path,
    message,
  );
}
