import { SimulatorError } from "./simulation-error.js";
import { allRanks, validateRankCount } from "./simulation-shared.js";
import type {
  RankGradientMap,
  DdpGradientSyncEvent,
  DdpGradientSynchronizationInput,
  DdpGradientSynchronizationResult,
  FsdpTensorShard,
  FsdpRankShardMap,
  FsdpParameterShardPlan,
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
  ShardedOptimizerEvent,
  ShardedAdamWResult,
} from "./simulation-types.js";

export function simulateShardedAdamWStep(input: ShardedAdamWInput): ShardedAdamWResult {
  const rankCount = validateRankCount(input.ranks);
  const options = normalizeAdamWOptions(input.optimizer);
  const ownership = assignOptimizerStateOwners(input.parameters, rankCount);
  const updatedParameters: Record<string, number[]> = {};
  const nextState: Record<string, ShardedAdamWState> = {};
  const events: ShardedOptimizerEvent[] = [];
  let eventStep = 0;

  for (const owner of ownership) {
    events.push({
      step: eventStep,
      kind: "optimizer-state-shard",
      participants: allRanks(rankCount),
      rank: owner.rank,
      parameters: [...owner.parameters],
    });
    eventStep += 1;
  }

  for (const parameter of input.parameters) {
    validateParameterName(parameter.name);
    const values = validateTensorValues(parameter.values, parameter.name);
    if (parameter.trainable === false) {
      updatedParameters[parameter.name] = values;
      continue;
    }
    const gradients = validateTensorValues(parameter.gradients, `${parameter.name}.grad`);
    if (gradients.length !== values.length) {
      throw new SimulatorError(
        `gradient length for ${parameter.name} must match parameter length`,
      );
    }
    const previous = normalizeAdamWState(
      input.state?.[parameter.name],
      values.length,
      parameter.name,
    );
    const stepped = adamWStep(values, gradients, previous, options);
    updatedParameters[parameter.name] = stepped.values;
    nextState[parameter.name] = stepped.state;
  }

  events.push({
    step: eventStep,
    kind: "sharded-adamw-step",
    participants: allRanks(rankCount),
  });

  return {
    updatedParameters: cloneFullParameterMap(updatedParameters),
    nextState: cloneAdamWStateMap(nextState),
    ownership: ownership.map(cloneShardedOptimizerOwnership),
    events: events.map(cloneShardedOptimizerEvent),
  };
}

export function simulateDdpGradientSynchronization(
  input: DdpGradientSynchronizationInput,
): DdpGradientSynchronizationResult {
  const rankCount = validateRankCount(input.rankGradients.length);
  const synchronizedGradients = Array.from({ length: rankCount }, () => ({})) as Array<
    Record<string, number[]>
  >;
  const events: DdpGradientSyncEvent[] = [];
  let step = 0;

  for (const parameter of input.parameters) {
    validateParameterName(parameter.name);
    if (parameter.trainable === false) continue;
    const rankValues = input.rankGradients.map((gradients, rank) =>
      readGradient(gradients, parameter.name, rank),
    );
    const outputGradient = averageGradients(rankValues, parameter.name);
    for (const rankGradients of synchronizedGradients) {
      rankGradients[parameter.name] = [...outputGradient];
    }
    events.push({
      step,
      kind: "ddp-gradient-all-reduce",
      parameter: parameter.name,
      participants: allRanks(rankCount),
      inputGradients: rankValues.map((values) => [...values]),
      outputGradient: [...outputGradient],
    });
    step += 1;
  }

  return {
    synchronizedGradients: synchronizedGradients.map(cloneRankGradientMap),
    events: events.map(cloneDdpGradientSyncEvent),
  };
}

export function simulateFsdpParameterSharding(
  input: FsdpParameterShardingInput,
): FsdpParameterShardingResult {
  const rankCount = validateRankCount(input.ranks);
  const rankShards = createEmptyRankShardMaps(rankCount);
  const fullParameters: Record<string, number[]> = {};
  const parameters: FsdpParameterShardPlan[] = [];
  const events: FsdpShardingEvent[] = [];
  let step = 0;

  for (const parameter of input.parameters) {
    validateParameterName(parameter.name);
    const values = validateTensorValues(parameter.values, parameter.name);
    const sharded = parameter.sharded !== false;
    const ranges: FsdpParameterShardPlan["ranges"] = sharded
      ? balancedRanges(values.length, rankCount).map((range, rank) => ({
          rank,
          start: range.start,
          end: range.end,
        }))
      : allRanks(rankCount).map((rank) => ({
          rank,
          start: 0,
          end: values.length,
          replicated: true,
        }));

    for (const range of ranges) {
      rankShards[range.rank]![parameter.name] = {
        start: range.start,
        end: range.end,
        values: values.slice(range.start, range.end),
        ...(range.replicated ? { replicated: true } : {}),
      };
    }
    fullParameters[parameter.name] = [...values];
    parameters.push({
      name: parameter.name,
      length: values.length,
      sharded,
      ranges,
    });
    events.push({
      step,
      kind: sharded ? "fsdp-shard" : "fsdp-replicate",
      parameter: parameter.name,
      participants: allRanks(rankCount),
      ranges: ranges.map(({ rank, start, end }) => ({ rank, start, end })),
    });
    step += 1;
  }

  events.push({
    step,
    kind: "fsdp-all-gather",
    participants: allRanks(rankCount),
  });

  return {
    shardPlan: {
      ranks: rankCount,
      parameters: parameters.map(cloneFsdpParameterShardPlan),
    },
    rankShards: rankShards.map(cloneFsdpRankShardMap),
    fullParameters: cloneFullParameterMap(fullParameters),
    events: events.map(cloneFsdpShardingEvent),
  };
}

export function simulateFsdpGradientReduceScatter(
  input: FsdpGradientReduceScatterInput,
): FsdpGradientReduceScatterResult {
  const rankCount = validateRankCount(input.shardPlan.ranks);
  if (input.rankGradients.length !== rankCount) {
    throw new SimulatorError("rankGradients length must match shardPlan ranks");
  }
  const rankGradientShards = createEmptyRankShardMaps(rankCount);
  const events: FsdpShardingEvent[] = [];
  let step = 0;

  for (const parameter of input.shardPlan.parameters) {
    const rankValues = input.rankGradients.map((gradients, rank) =>
      readGradient(gradients, parameter.name, rank),
    );
    const averaged = averageGradients(rankValues, parameter.name);
    if (averaged.length !== parameter.length) {
      throw new SimulatorError(
        `gradient length for parameter ${parameter.name} does not match shard plan`,
      );
    }
    for (const range of parameter.ranges) {
      rankGradientShards[range.rank]![parameter.name] = {
        start: range.start,
        end: range.end,
        values: averaged.slice(range.start, range.end),
        ...(range.replicated ? { replicated: true } : {}),
      };
    }
    events.push({
      step,
      kind: parameter.sharded
        ? "fsdp-reduce-scatter"
        : "fsdp-replicated-gradient-all-reduce",
      parameter: parameter.name,
      participants: allRanks(rankCount),
      ranges: parameter.ranges.map(({ rank, start, end }) => ({ rank, start, end })),
      output: [...averaged],
    });
    step += 1;
  }

  return {
    rankGradientShards: rankGradientShards.map(cloneFsdpRankShardMap),
    events: events.map(cloneFsdpShardingEvent),
  };
}

function validateParameterName(name: string): void {
  if (name.length === 0) {
    throw new SimulatorError("parameter name must be non-empty");
  }
}

function readGradient(
  gradients: RankGradientMap,
  parameter: string,
  rank: number,
): readonly number[] {
  const value = gradients[parameter];
  if (!value) {
    throw new SimulatorError(
      `rank ${rank} is missing gradient for parameter ${parameter}`,
    );
  }
  return value.map((entry, index) => {
    if (!Number.isFinite(entry)) {
      throw new SimulatorError(
        `gradient ${parameter}[${index}] on rank ${rank} must be finite`,
      );
    }
    return entry;
  });
}

function averageGradients(
  gradients: readonly (readonly number[])[],
  parameter: string,
): number[] {
  const width = gradients[0]?.length;
  if (width === undefined) {
    throw new SimulatorError(`parameter ${parameter} has no gradient values`);
  }
  for (let rank = 0; rank < gradients.length; rank++) {
    if ((gradients[rank]?.length ?? -1) !== width) {
      throw new SimulatorError(
        `gradient shape mismatch for parameter ${parameter} on rank ${rank}`,
      );
    }
  }
  return Array.from({ length: width }, (_, index) => {
    const sum = gradients.reduce((total, values) => total + (values[index] ?? 0), 0);
    return sum / gradients.length;
  });
}

function cloneRankGradientMap(map: Record<string, number[]>): RankGradientMap {
  const out: Record<string, number[]> = {};
  for (const [name, values] of Object.entries(map)) {
    out[name] = [...values];
  }
  return out;
}

function cloneDdpGradientSyncEvent(event: DdpGradientSyncEvent): DdpGradientSyncEvent {
  return {
    ...event,
    participants: [...event.participants],
    inputGradients: event.inputGradients.map((gradient) => [...gradient]),
    outputGradient: [...event.outputGradient],
  };
}

function validateTensorValues(values: readonly number[], name: string): number[] {
  if (values.length === 0) {
    throw new SimulatorError(`parameter ${name} must have at least one value`);
  }
  return values.map((value, index) => {
    if (!Number.isFinite(value)) {
      throw new SimulatorError(`parameter ${name}[${index}] must be finite`);
    }
    return value;
  });
}

function normalizeAdamWOptions(options: ShardedAdamWOptions): Required<ShardedAdamWOptions> {
  const lr = finitePositiveNumber(options.lr, "lr");
  const weightDecay = finiteNonNegativeNumber(options.weightDecay ?? 0, "weightDecay");
  const betas = options.betas ?? [0.9, 0.999];
  const beta1 = finiteBeta(betas[0], "beta1");
  const beta2 = finiteBeta(betas[1], "beta2");
  const eps = finiteNonNegativeNumber(options.eps ?? 1e-8, "eps");
  return { lr, weightDecay, betas: [beta1, beta2], eps };
}

function assignOptimizerStateOwners(
  parameters: readonly ShardedAdamWParameterSpec[],
  rankCount: number,
): ShardedOptimizerOwnership[] {
  const ownership = allRanks(rankCount).map((rank) => ({
    rank,
    parameters: [] as string[],
    elements: 0,
  }));
  for (const parameter of parameters) {
    validateParameterName(parameter.name);
    if (parameter.trainable === false) continue;
    const values = validateTensorValues(parameter.values, parameter.name);
    const owner = ownership.reduce((best, candidate) => {
      if (candidate.elements < best.elements) return candidate;
      if (candidate.elements === best.elements && candidate.rank < best.rank) {
        return candidate;
      }
      return best;
    });
    owner.parameters.push(parameter.name);
    owner.elements += values.length;
  }
  return ownership;
}

function normalizeAdamWState(
  state: ShardedAdamWState | undefined,
  length: number,
  parameter: string,
): ShardedAdamWState {
  if (!state) {
    return {
      step: 0,
      expAvg: Array.from({ length }, () => 0),
      expAvgSq: Array.from({ length }, () => 0),
    };
  }
  if (!Number.isInteger(state.step) || state.step < 0) {
    throw new SimulatorError(`AdamW state step for ${parameter} must be non-negative`);
  }
  if (state.expAvg.length !== length || state.expAvgSq.length !== length) {
    throw new SimulatorError(`AdamW state for ${parameter} has wrong shape`);
  }
  return {
    step: state.step,
    expAvg: validateTensorValues(state.expAvg, `${parameter}.expAvg`),
    expAvgSq: validateTensorValues(state.expAvgSq, `${parameter}.expAvgSq`),
  };
}

function adamWStep(
  values: readonly number[],
  gradients: readonly number[],
  previous: ShardedAdamWState,
  options: Required<ShardedAdamWOptions>,
): { values: number[]; state: ShardedAdamWState } {
  const [beta1, beta2] = options.betas;
  const step = previous.step + 1;
  const biasCorrection1 = 1 - beta1 ** step;
  const biasCorrection2 = 1 - beta2 ** step;
  const nextValues: number[] = [];
  const expAvg: number[] = [];
  const expAvgSq: number[] = [];

  for (let index = 0; index < values.length; index++) {
    const gradient = gradients[index] ?? 0;
    const nextExpAvg = beta1 * (previous.expAvg[index] ?? 0) + (1 - beta1) * gradient;
    const nextExpAvgSq =
      beta2 * (previous.expAvgSq[index] ?? 0) + (1 - beta2) * gradient ** 2;
    const correctedExpAvg = nextExpAvg / biasCorrection1;
    const correctedExpAvgSq = nextExpAvgSq / biasCorrection2;
    const decayedValue = (values[index] ?? 0) * (1 - options.lr * options.weightDecay);
    nextValues.push(
      decayedValue -
        (options.lr * correctedExpAvg) / (Math.sqrt(correctedExpAvgSq) + options.eps),
    );
    expAvg.push(nextExpAvg);
    expAvgSq.push(nextExpAvgSq);
  }

  return {
    values: nextValues,
    state: { step, expAvg, expAvgSq },
  };
}

function finitePositiveNumber(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new SimulatorError(`${name} must be positive`);
  }
  return value;
}

function finiteNonNegativeNumber(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new SimulatorError(`${name} must be non-negative`);
  }
  return value;
}

function finiteBeta(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0 || value >= 1) {
    throw new SimulatorError(`${name} must be in [0, 1)`);
  }
  return value;
}

function cloneAdamWStateMap(
  map: Record<string, ShardedAdamWState>,
): Readonly<Record<string, ShardedAdamWState>> {
  const out: Record<string, ShardedAdamWState> = {};
  for (const [name, state] of Object.entries(map)) {
    out[name] = {
      step: state.step,
      expAvg: [...state.expAvg],
      expAvgSq: [...state.expAvgSq],
    };
  }
  return out;
}

function cloneShardedOptimizerOwnership(
  ownership: ShardedOptimizerOwnership,
): ShardedOptimizerOwnership {
  return {
    rank: ownership.rank,
    parameters: [...ownership.parameters],
    elements: ownership.elements,
  };
}

function cloneShardedOptimizerEvent(event: ShardedOptimizerEvent): ShardedOptimizerEvent {
  return {
    ...event,
    participants: [...event.participants],
    ...(event.parameters ? { parameters: [...event.parameters] } : {}),
  };
}

function balancedRanges(
  length: number,
  rankCount: number,
): Array<{ start: number; end: number }> {
  const base = Math.floor(length / rankCount);
  const remainder = length % rankCount;
  const ranges: Array<{ start: number; end: number }> = [];
  let start = 0;
  for (let rank = 0; rank < rankCount; rank++) {
    const width = base + (rank < remainder ? 1 : 0);
    ranges.push({ start, end: start + width });
    start += width;
  }
  return ranges;
}

function createEmptyRankShardMaps(rankCount: number): Array<Record<string, FsdpTensorShard>> {
  return Array.from({ length: rankCount }, () => ({}));
}

function cloneFsdpRankShardMap(map: Record<string, FsdpTensorShard>): FsdpRankShardMap {
  const out: Record<string, FsdpTensorShard> = {};
  for (const [name, shard] of Object.entries(map)) {
    out[name] = {
      start: shard.start,
      end: shard.end,
      values: [...shard.values],
      ...(shard.replicated ? { replicated: true } : {}),
    };
  }
  return out;
}

function cloneFullParameterMap(
  map: Record<string, readonly number[]>,
): Readonly<Record<string, readonly number[]>> {
  const out: Record<string, number[]> = {};
  for (const [name, values] of Object.entries(map)) {
    out[name] = [...values];
  }
  return out;
}

function cloneFsdpParameterShardPlan(
  plan: FsdpParameterShardPlan,
): FsdpParameterShardPlan {
  return {
    ...plan,
    ranges: plan.ranges.map((range) => ({ ...range })),
  };
}

function cloneFsdpShardingEvent(event: FsdpShardingEvent): FsdpShardingEvent {
  return {
    ...event,
    participants: [...event.participants],
    ...(event.ranges ? { ranges: event.ranges.map((range) => ({ ...range })) } : {}),
    ...(event.output ? { output: [...event.output] } : {}),
  };
}
