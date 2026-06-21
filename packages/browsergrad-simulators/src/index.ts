export type SimulationEventKind =
  | "barrier"
  | "send"
  | "deliver"
  | "all-reduce";

export type SimulationReductionOp = "sum" | "min" | "max";

export interface SimulationEvent {
  readonly step: number;
  readonly kind: SimulationEventKind;
  readonly tag: string;
  readonly from?: number;
  readonly to?: number;
  readonly participants?: number[];
  readonly payload?: unknown;
  readonly op?: SimulationReductionOp;
  readonly values?: readonly number[];
  readonly result?: number;
}

export interface DeterministicMeshOptions {
  readonly ranks: number;
}

export interface MeshSendOptions {
  readonly from: number;
  readonly to: number;
  readonly tag: string;
  readonly payload?: unknown;
}

export interface MeshBroadcastOptions {
  readonly from: number;
  readonly tag: string;
  readonly payload?: unknown;
}

export interface MeshAllReduceOptions {
  readonly tag: string;
  readonly values: readonly number[];
  readonly op: SimulationReductionOp;
}

export interface DeterministicMesh {
  readonly rankCount: number;
  barrier(tag?: string): void;
  send(message: MeshSendOptions): void;
  broadcast(message: MeshBroadcastOptions): void;
  allReduce(options: MeshAllReduceOptions): number[];
  trace(): SimulationEvent[];
  clear(): void;
}

export type TaskGraphEventKind =
  | "task-ready"
  | "task-start"
  | "task-finish";

export interface TaskSpec {
  readonly id: string;
  readonly duration: number;
  readonly dependsOn?: readonly string[];
}

export interface TaskGraphEvent {
  time: number;
  kind: TaskGraphEventKind;
  taskId: string;
  worker?: number;
}

export interface TaskGraphRunResult {
  readonly makespan: number;
  readonly completedTaskIds: string[];
  readonly events: TaskGraphEvent[];
}

export interface TaskGraphSimulatorOptions {
  readonly workers: number;
}

export interface TaskGraphSimulator {
  readonly workerCount: number;
  addTask(task: TaskSpec): void;
  run(): TaskGraphRunResult;
  clear(): void;
}

export interface DistributedParameterSpec {
  readonly name: string;
  readonly trainable?: boolean;
}

export type RankGradientMap = Readonly<Record<string, readonly number[]>>;

export interface DdpGradientSyncEvent {
  readonly step: number;
  readonly kind: "ddp-gradient-all-reduce";
  readonly parameter: string;
  readonly participants: readonly number[];
  readonly inputGradients: readonly (readonly number[])[];
  readonly outputGradient: readonly number[];
}

export interface DdpGradientSynchronizationInput {
  readonly parameters: readonly DistributedParameterSpec[];
  readonly rankGradients: readonly RankGradientMap[];
}

export interface DdpGradientSynchronizationResult {
  readonly synchronizedGradients: readonly RankGradientMap[];
  readonly events: readonly DdpGradientSyncEvent[];
}

export interface DistributedParameterValueSpec {
  readonly name: string;
  readonly values: readonly number[];
  readonly sharded?: boolean;
}

export interface FsdpTensorShard {
  readonly start: number;
  readonly end: number;
  readonly values: readonly number[];
  readonly replicated?: boolean;
}

export type FsdpRankShardMap = Readonly<Record<string, FsdpTensorShard>>;

export interface FsdpParameterShardPlan {
  readonly name: string;
  readonly length: number;
  readonly sharded: boolean;
  readonly ranges: readonly {
    readonly rank: number;
    readonly start: number;
    readonly end: number;
    readonly replicated?: boolean;
  }[];
}

export interface FsdpShardPlan {
  readonly ranks: number;
  readonly parameters: readonly FsdpParameterShardPlan[];
}

export type FsdpShardingEventKind =
  | "fsdp-shard"
  | "fsdp-replicate"
  | "fsdp-all-gather"
  | "fsdp-reduce-scatter"
  | "fsdp-replicated-gradient-all-reduce";

export interface FsdpShardingEvent {
  readonly step: number;
  readonly kind: FsdpShardingEventKind;
  readonly parameter?: string;
  readonly participants: readonly number[];
  readonly ranges?: readonly {
    readonly rank: number;
    readonly start: number;
    readonly end: number;
  }[];
  readonly output?: readonly number[];
}

export interface FsdpParameterShardingInput {
  readonly ranks: number;
  readonly parameters: readonly DistributedParameterValueSpec[];
}

export interface FsdpParameterShardingResult {
  readonly shardPlan: FsdpShardPlan;
  readonly rankShards: readonly FsdpRankShardMap[];
  readonly fullParameters: Readonly<Record<string, readonly number[]>>;
  readonly events: readonly FsdpShardingEvent[];
}

export interface FsdpGradientReduceScatterInput {
  readonly shardPlan: FsdpShardPlan;
  readonly rankGradients: readonly RankGradientMap[];
}

export interface FsdpGradientReduceScatterResult {
  readonly rankGradientShards: readonly FsdpRankShardMap[];
  readonly events: readonly FsdpShardingEvent[];
}

export interface ShardedAdamWParameterSpec {
  readonly name: string;
  readonly values: readonly number[];
  readonly gradients: readonly number[];
  readonly trainable?: boolean;
}

export interface ShardedAdamWOptions {
  readonly lr: number;
  readonly weightDecay?: number;
  readonly betas?: readonly [number, number];
  readonly eps?: number;
}

export interface ShardedAdamWState {
  readonly step: number;
  readonly expAvg: readonly number[];
  readonly expAvgSq: readonly number[];
}

export interface ShardedAdamWInput {
  readonly ranks: number;
  readonly parameters: readonly ShardedAdamWParameterSpec[];
  readonly optimizer: ShardedAdamWOptions;
  readonly state?: Readonly<Record<string, ShardedAdamWState>>;
}

export interface ShardedOptimizerOwnership {
  readonly rank: number;
  readonly parameters: readonly string[];
  readonly elements: number;
}

export type ShardedOptimizerEventKind =
  | "optimizer-state-shard"
  | "sharded-adamw-step";

export interface ShardedOptimizerEvent {
  readonly step: number;
  readonly kind: ShardedOptimizerEventKind;
  readonly participants: readonly number[];
  readonly rank?: number;
  readonly parameters?: readonly string[];
}

export interface ShardedAdamWResult {
  readonly updatedParameters: Readonly<Record<string, readonly number[]>>;
  readonly nextState: Readonly<Record<string, ShardedAdamWState>>;
  readonly ownership: readonly ShardedOptimizerOwnership[];
  readonly events: readonly ShardedOptimizerEvent[];
}

export class SimulatorError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SimulatorError";
  }
}

export function createDeterministicMesh(
  options: DeterministicMeshOptions,
): DeterministicMesh {
  const rankCount = validateRankCount(options.ranks);
  const events: SimulationEvent[] = [];
  let step = 0;

  const push = (event: Omit<SimulationEvent, "step">): void => {
    events.push({ step, ...cloneEvent(event) });
    step += 1;
  };

  const validateRank = (rank: number, field: string): void => {
    if (!Number.isInteger(rank) || rank < 0 || rank >= rankCount) {
      throw new SimulatorError(`${field} must be an integer rank in [0, ${rankCount})`);
    }
  };

  const send = (message: MeshSendOptions): void => {
    validateRank(message.from, "from");
    validateRank(message.to, "to");
    if (message.from === message.to) {
      throw new SimulatorError("send requires distinct from/to ranks");
    }
    push({
      kind: "send",
      tag: message.tag,
      from: message.from,
      to: message.to,
      ...(message.payload !== undefined ? { payload: message.payload } : {}),
    });
    push({
      kind: "deliver",
      tag: message.tag,
      from: message.from,
      to: message.to,
      ...(message.payload !== undefined ? { payload: message.payload } : {}),
    });
  };

  return {
    rankCount,
    barrier(tag = "barrier") {
      push({
        kind: "barrier",
        tag,
        participants: allRanks(rankCount),
      });
    },
    send,
    broadcast(message) {
      validateRank(message.from, "from");
      for (const to of allRanks(rankCount)) {
        if (to === message.from) continue;
        send({
          from: message.from,
          to,
          tag: message.tag,
          ...(message.payload !== undefined ? { payload: message.payload } : {}),
        });
      }
    },
    allReduce(options) {
      if (options.values.length !== rankCount) {
        throw new SimulatorError(
          `allReduce values length must equal rank count ${rankCount}`,
        );
      }
      for (let i = 0; i < options.values.length; i++) {
        const value = options.values[i];
        if (typeof value !== "number" || !Number.isFinite(value)) {
          throw new SimulatorError(`allReduce values[${i}] must be finite`);
        }
      }
      const result = reduceValues(options.values, options.op);
      push({
        kind: "all-reduce",
        tag: options.tag,
        op: options.op,
        participants: allRanks(rankCount),
        values: [...options.values],
        result,
      });
      return Array.from({ length: rankCount }, () => result);
    },
    trace() {
      return events.map(cloneEvent);
    },
    clear() {
      events.length = 0;
      step = 0;
    },
  };
}

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

function validateRankCount(ranks: number): number {
  if (!Number.isInteger(ranks) || ranks <= 0) {
    throw new SimulatorError("ranks must be a positive integer");
  }
  return ranks;
}

function allRanks(rankCount: number): number[] {
  return Array.from({ length: rankCount }, (_, rank) => rank);
}

function reduceValues(
  values: readonly number[],
  op: SimulationReductionOp,
): number {
  if (op === "sum") {
    return values.reduce((acc, value) => acc + value, 0);
  }
  if (op === "min") {
    return Math.min(...values);
  }
  if (op === "max") {
    return Math.max(...values);
  }
  throw new SimulatorError(`unsupported reduction op: ${String(op)}`);
}

function cloneEvent<T extends Omit<SimulationEvent, "step"> | SimulationEvent>(
  event: T,
): T {
  return {
    ...event,
    ...(event.participants ? { participants: [...event.participants] } : {}),
    ...(event.values ? { values: [...event.values] } : {}),
    ...(event.payload !== undefined ? { payload: clonePayload(event.payload) } : {}),
  };
}

function clonePayload(payload: unknown): unknown {
  if (payload === null || typeof payload !== "object") return payload;
  if (Array.isArray(payload)) return payload.map(clonePayload);
  if (payload instanceof Uint8Array) return new Uint8Array(payload);
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(payload)) {
    out[key] = clonePayload(value);
  }
  return out;
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

export function createTaskGraphSimulator(
  options: TaskGraphSimulatorOptions,
): TaskGraphSimulator {
  const workerCount = validateWorkerCount(options.workers);
  const tasks = new Map<string, Required<TaskSpec>>();

  return {
    workerCount,
    addTask(task) {
      const normalized = normalizeTask(task);
      if (tasks.has(normalized.id)) {
        throw new SimulatorError(`duplicate task id: ${normalized.id}`);
      }
      tasks.set(normalized.id, normalized);
    },
    run() {
      return runTaskGraph([...tasks.values()], workerCount);
    },
    clear() {
      tasks.clear();
    },
  };
}

function validateWorkerCount(workers: number): number {
  if (!Number.isInteger(workers) || workers <= 0) {
    throw new SimulatorError("workers must be a positive integer");
  }
  return workers;
}

function normalizeTask(task: TaskSpec): Required<TaskSpec> {
  if (task.id.length === 0) {
    throw new SimulatorError("task id must be non-empty");
  }
  if (!Number.isFinite(task.duration) || task.duration <= 0) {
    throw new SimulatorError(`task duration must be positive: ${task.id}`);
  }
  return {
    id: task.id,
    duration: task.duration,
    dependsOn: uniqueSorted(task.dependsOn ?? []),
  };
}

function runTaskGraph(
  tasks: readonly Required<TaskSpec>[],
  workerCount: number,
): TaskGraphRunResult {
  const taskById = new Map(tasks.map((task) => [task.id, task] as const));
  validateTaskDependencies(tasks, taskById);

  const events: TaskGraphEvent[] = [];
  const completed = new Set<string>();
  const started = new Set<string>();
  const announcedReady = new Set<string>();
  const completedTaskIds: string[] = [];
  const workers = Array.from({ length: workerCount }, (_, worker) => ({
    worker,
    taskId: undefined as string | undefined,
    finishTime: Number.POSITIVE_INFINITY,
  }));
  let time = 0;

  const emit = (event: TaskGraphEvent): void => {
    events.push(cloneTaskGraphEvent(event));
  };

  while (completed.size < tasks.length) {
    const ready = readyTasks(tasks, completed, started);
    for (const task of ready) {
      if (!announcedReady.has(task.id)) {
        emit({ time, kind: "task-ready", taskId: task.id });
        announcedReady.add(task.id);
      }
    }

    let startedThisTick = false;
    for (const worker of workers.filter((item) => item.taskId === undefined)) {
      const next = ready.find((task) => !started.has(task.id));
      if (!next) break;
      started.add(next.id);
      worker.taskId = next.id;
      worker.finishTime = time + next.duration;
      emit({
        time,
        kind: "task-start",
        taskId: next.id,
        worker: worker.worker,
      });
      startedThisTick = true;
    }
    if (startedThisTick) continue;

    const nextFinishTime = Math.min(...workers.map((worker) => worker.finishTime));
    if (!Number.isFinite(nextFinishTime)) {
      throw new SimulatorError("task graph has a dependency cycle");
    }
    time = nextFinishTime;

    for (const worker of workers
      .filter((item) => item.finishTime === time && item.taskId !== undefined)
      .sort((a, b) => a.worker - b.worker)) {
      const taskId = worker.taskId!;
      worker.taskId = undefined;
      worker.finishTime = Number.POSITIVE_INFINITY;
      completed.add(taskId);
      completedTaskIds.push(taskId);
      emit({ time, kind: "task-finish", taskId, worker: worker.worker });
    }
  }

  return {
    makespan: time,
    completedTaskIds: [...completedTaskIds],
    events: events.map(cloneTaskGraphEvent),
  };
}

function validateTaskDependencies(
  tasks: readonly Required<TaskSpec>[],
  taskById: ReadonlyMap<string, Required<TaskSpec>>,
): void {
  for (const task of tasks) {
    for (const dependency of task.dependsOn) {
      if (!taskById.has(dependency)) {
        throw new SimulatorError(
          `task ${task.id} depends on missing task: ${dependency}`,
        );
      }
    }
  }
}

function readyTasks(
  tasks: readonly Required<TaskSpec>[],
  completed: ReadonlySet<string>,
  started: ReadonlySet<string>,
): Required<TaskSpec>[] {
  return tasks
    .filter(
      (task) =>
        !completed.has(task.id) &&
        !started.has(task.id) &&
        task.dependsOn.every((dependency) => completed.has(dependency)),
    )
    .sort((a, b) => a.id.localeCompare(b.id));
}

function cloneTaskGraphEvent(event: TaskGraphEvent): TaskGraphEvent {
  return { ...event };
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort();
}
