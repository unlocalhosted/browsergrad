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
