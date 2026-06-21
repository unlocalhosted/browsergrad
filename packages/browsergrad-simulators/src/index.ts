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
