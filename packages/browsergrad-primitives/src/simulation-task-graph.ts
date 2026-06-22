import { SimulatorError } from "./simulation-error.js";
import type {
  TaskSpec,
  TaskGraphEvent,
  TaskGraphRunResult,
  TaskGraphSimulatorOptions,
  TaskGraphSimulator,
} from "./simulation-types.js";

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
