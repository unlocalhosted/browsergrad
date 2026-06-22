import { SimulatorError } from "./simulation-error.js";
import { validateRankCount } from "./simulation-shared.js";
import type {
  SimdInstructionOp,
  SimdTraceEvent,
  VectorizedClampedExpInput,
  VectorizedClampedExpResult,
  VectorizedArraySumInput,
  VectorizedArraySumResult,
  StaticWorkRange,
  StaticWorkPartition,
  StaticWorkPartitionInput,
} from "./simulation-types.js";

export { SimulatorError } from "./simulation-error.js";
export { createDeterministicMesh } from "./simulation-mesh.js";
export { createTaskGraphSimulator } from "./simulation-task-graph.js";
export {
  simulateDdpGradientSynchronization,
  simulateFsdpGradientReduceScatter,
  simulateFsdpParameterSharding,
  simulateShardedAdamWStep,
} from "./simulation-distributed.js";
export type {
  SimulationEventKind,
  SimulationReductionOp,
  SimulationEvent,
  DeterministicMeshOptions,
  MeshSendOptions,
  MeshBroadcastOptions,
  MeshAllReduceOptions,
  DeterministicMesh,
  TaskGraphEventKind,
  TaskSpec,
  TaskGraphEvent,
  TaskGraphRunResult,
  TaskGraphSimulatorOptions,
  TaskGraphSimulator,
  SimdInstructionOp,
  SimdTraceEvent,
  SimdKernelStats,
  VectorizedClampedExpInput,
  VectorizedClampedExpResult,
  VectorizedArraySumInput,
  VectorizedArraySumResult,
  StaticWorkRange,
  StaticWorkPartition,
  StaticWorkPartitionInput,
  DistributedParameterSpec,
  RankGradientMap,
  DdpGradientSyncEvent,
  DdpGradientSynchronizationInput,
  DdpGradientSynchronizationResult,
  DistributedParameterValueSpec,
  FsdpTensorShard,
  FsdpRankShardMap,
  FsdpParameterShardPlan,
  FsdpShardPlan,
  FsdpShardingEventKind,
  FsdpShardingEvent,
  FsdpParameterShardingInput,
  FsdpParameterShardingResult,
  FsdpGradientReduceScatterInput,
  FsdpGradientReduceScatterResult,
  ShardedAdamWParameterSpec,
  ShardedAdamWOptions,
  ShardedAdamWState,
  ShardedAdamWInput,
  ShardedOptimizerOwnership,
  ShardedOptimizerEventKind,
  ShardedOptimizerEvent,
  ShardedAdamWResult,
} from "./simulation-types.js";

export function simulateVectorizedClampedExp(
  input: VectorizedClampedExpInput,
): VectorizedClampedExpResult {
  const vectorWidth = validateVectorWidth(input.vectorWidth);
  const values = validateNumberVector(input.values, "values");
  if (values.length === 0) {
    throw new SimulatorError("values must contain at least one lane");
  }
  const exponents = validateExponentVector(input.exponents);
  if (exponents.length !== values.length) {
    throw new SimulatorError("values and exponents must have the same length");
  }
  const clamp = finitePositiveNumber(input.clamp ?? 9.999999, "clamp");
  const chunks = Math.ceil(values.length / vectorWidth);
  const laneSlots = chunks * vectorWidth;
  const output = Array.from({ length: values.length }, () => 0);
  const trace: SimdTraceEvent[] = [];
  let step = 0;
  let activeLaneTotal = 0;
  let laneTotal = 0;

  const record = (
    op: SimdInstructionOp,
    chunk: number,
    laneStart: number,
    mask: readonly boolean[],
  ): void => {
    const activeLanes = countActive(mask);
    trace.push({
      step,
      op,
      chunk,
      laneStart,
      mask: [...mask],
      activeLanes,
    });
    step += 1;
    activeLaneTotal += activeLanes;
    laneTotal += vectorWidth;
  };

  for (let chunk = 0; chunk < chunks; chunk++) {
    const laneStart = chunk * vectorWidth;
    const validMask = Array.from(
      { length: vectorWidth },
      (_, lane) => laneStart + lane < values.length,
    );
    const result = Array.from({ length: vectorWidth }, () => 1);
    const remaining = Array.from({ length: vectorWidth }, (_, lane) => {
      const index = laneStart + lane;
      return validMask[lane] ? exponents[index]! : 0;
    });

    record("load-values", chunk, laneStart, validMask);
    record("load-exponents", chunk, laneStart, validMask);

    while (remaining.some((count, lane) => validMask[lane] === true && count > 0)) {
      const activeMask = remaining.map(
        (count, lane) => validMask[lane] === true && count > 0,
      );
      record("mul", chunk, laneStart, activeMask);
      for (let lane = 0; lane < vectorWidth; lane++) {
        if (!activeMask[lane]) continue;
        result[lane]! *= values[laneStart + lane]!;
        remaining[lane]! -= 1;
      }
      record("decrement-exponents", chunk, laneStart, activeMask);
    }

    const clampMask = result.map(
      (value, lane) => validMask[lane] === true && value > clamp,
    );
    record("clamp", chunk, laneStart, clampMask);
    for (let lane = 0; lane < vectorWidth; lane++) {
      const index = laneStart + lane;
      if (!validMask[lane] || index >= output.length) continue;
      output[index] = clampMask[lane] ? clamp : result[lane]!;
    }
    record("store", chunk, laneStart, validMask);
  }

  return {
    output: [...output],
    stats: {
      vectorWidth,
      chunks,
      inputLanes: values.length,
      laneSlots,
      tailInactiveLanes: laneSlots - values.length,
      vectorInstructions: trace.length,
      activeLanes: activeLaneTotal,
      totalLanes: laneTotal,
      utilization: laneTotal === 0 ? 0 : activeLaneTotal / laneTotal,
    },
    trace: trace.map(cloneSimdTraceEvent),
  };
}

export function simulateVectorizedArraySum(
  input: VectorizedArraySumInput,
): VectorizedArraySumResult {
  const vectorWidth = validateVectorWidth(input.vectorWidth);
  const values = validateNumberVector(input.values, "values");
  if (values.length === 0) {
    throw new SimulatorError("values must contain at least one lane");
  }
  if (values.length % vectorWidth !== 0) {
    throw new SimulatorError(
      "arraySumVector fixtures must have values.length divisible by vectorWidth",
    );
  }
  const chunks = values.length / vectorWidth;
  const laneSlots = values.length;
  const partialLaneSums = Array.from({ length: vectorWidth }, () => 0);
  const trace: SimdTraceEvent[] = [];
  let step = 0;
  let activeLaneTotal = 0;
  let laneTotal = 0;

  const record = (
    op: SimdInstructionOp,
    chunk: number,
    laneStart: number,
    mask: readonly boolean[],
  ): void => {
    const activeLanes = countActive(mask);
    trace.push({
      step,
      op,
      chunk,
      laneStart,
      mask: [...mask],
      activeLanes,
    });
    step += 1;
    activeLaneTotal += activeLanes;
    laneTotal += vectorWidth;
  };

  const fullMask = Array.from({ length: vectorWidth }, () => true);
  for (let chunk = 0; chunk < chunks; chunk++) {
    const laneStart = chunk * vectorWidth;
    record("load-values", chunk, laneStart, fullMask);
    record("add-accumulator", chunk, laneStart, fullMask);
    for (let lane = 0; lane < vectorWidth; lane++) {
      partialLaneSums[lane]! += values[laneStart + lane]!;
    }
  }

  let working = [...partialLaneSums];
  let horizontalReductionRounds = 0;
  while (working.length > 1) {
    const active = working.length;
    record(
      "horizontal-add",
      chunks,
      0,
      Array.from({ length: vectorWidth }, (_, lane) => lane < active),
    );
    const next: number[] = [];
    for (let lane = 0; lane < working.length; lane += 2) {
      next.push(working[lane]! + (working[lane + 1] ?? 0));
    }
    working = next;
    horizontalReductionRounds += 1;
  }

  return {
    sum: working[0] ?? 0,
    partialLaneSums: [...partialLaneSums],
    horizontalReductionRounds,
    stats: {
      vectorWidth,
      chunks,
      inputLanes: values.length,
      laneSlots,
      tailInactiveLanes: 0,
      vectorInstructions: trace.length,
      activeLanes: activeLaneTotal,
      totalLanes: laneTotal,
      utilization: laneTotal === 0 ? 0 : activeLaneTotal / laneTotal,
    },
    trace: trace.map(cloneSimdTraceEvent),
  };
}

export function partitionStaticWork(
  input: StaticWorkPartitionInput,
): StaticWorkPartition[] {
  const itemCount = validateItemCount(input.items);
  const workerCount = validateRankCount(input.workers);
  const partitions = Array.from({ length: workerCount }, (_, worker) => ({
    worker,
    ranges: [] as StaticWorkRange[],
  }));

  if (input.chunkSize !== undefined) {
    const chunkSize = validateChunkSize(input.chunkSize);
    let chunkIndex = 0;
    for (let start = 0; start < itemCount; start += chunkSize) {
      const worker = chunkIndex % workerCount;
      partitions[worker]!.ranges.push({
        start,
        end: Math.min(start + chunkSize, itemCount),
      });
      chunkIndex += 1;
    }
    return partitions.map(cloneStaticWorkPartition);
  }

  const base = Math.floor(itemCount / workerCount);
  const remainder = itemCount % workerCount;
  let start = 0;
  for (let worker = 0; worker < workerCount; worker++) {
    const size = base + (worker < remainder ? 1 : 0);
    if (size > 0) {
      partitions[worker]!.ranges.push({
        start,
        end: start + size,
      });
    }
    start += size;
  }
  return partitions.map(cloneStaticWorkPartition);
}

function validateVectorWidth(width: number): number {
  if (!Number.isInteger(width) || width <= 0) {
    throw new SimulatorError("vectorWidth must be a positive integer");
  }
  return width;
}

function validateNumberVector(values: readonly number[], name: string): number[] {
  return values.map((value, index) => {
    if (!Number.isFinite(value)) {
      throw new SimulatorError(`${name}[${index}] must be finite`);
    }
    return value;
  });
}

function validateExponentVector(exponents: readonly number[]): number[] {
  return exponents.map((value, index) => {
    if (!Number.isInteger(value) || value < 0) {
      throw new SimulatorError(`exponents[${index}] must be a non-negative integer`);
    }
    return value;
  });
}

function countActive(mask: readonly boolean[]): number {
  return mask.reduce((count, active) => count + (active ? 1 : 0), 0);
}

function validateItemCount(items: number): number {
  if (!Number.isInteger(items) || items < 0) {
    throw new SimulatorError("items must be a non-negative integer");
  }
  return items;
}

function validateChunkSize(chunkSize: number): number {
  if (!Number.isInteger(chunkSize) || chunkSize <= 0) {
    throw new SimulatorError("chunkSize must be a positive integer");
  }
  return chunkSize;
}

function cloneSimdTraceEvent(
  event: SimdTraceEvent,
): SimdTraceEvent {
  return {
    ...event,
    mask: [...event.mask],
  };
}

function cloneStaticWorkPartition(partition: StaticWorkPartition): StaticWorkPartition {
  return {
    worker: partition.worker,
    ranges: partition.ranges.map((range) => ({ ...range })),
  };
}

function finitePositiveNumber(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new SimulatorError(`${name} must be positive`);
  }
  return value;
}
