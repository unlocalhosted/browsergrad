export class SimulatorError extends Error {
    constructor(message) {
        super(message);
        this.name = "SimulatorError";
    }
}
export function createDeterministicMesh(options) {
    const rankCount = validateRankCount(options.ranks);
    const events = [];
    let step = 0;
    const push = (event) => {
        events.push({ step, ...cloneEvent(event) });
        step += 1;
    };
    const validateRank = (rank, field) => {
        if (!Number.isInteger(rank) || rank < 0 || rank >= rankCount) {
            throw new SimulatorError(`${field} must be an integer rank in [0, ${rankCount})`);
        }
    };
    const send = (message) => {
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
                if (to === message.from)
                    continue;
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
                throw new SimulatorError(`allReduce values length must equal rank count ${rankCount}`);
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
function validateRankCount(ranks) {
    if (!Number.isInteger(ranks) || ranks <= 0) {
        throw new SimulatorError("ranks must be a positive integer");
    }
    return ranks;
}
function allRanks(rankCount) {
    return Array.from({ length: rankCount }, (_, rank) => rank);
}
function reduceValues(values, op) {
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
function cloneEvent(event) {
    return {
        ...event,
        ...(event.participants ? { participants: [...event.participants] } : {}),
        ...(event.values ? { values: [...event.values] } : {}),
        ...(event.payload !== undefined ? { payload: clonePayload(event.payload) } : {}),
    };
}
function clonePayload(payload) {
    if (payload === null || typeof payload !== "object")
        return payload;
    if (Array.isArray(payload))
        return payload.map(clonePayload);
    if (payload instanceof Uint8Array)
        return new Uint8Array(payload);
    const out = {};
    for (const [key, value] of Object.entries(payload)) {
        out[key] = clonePayload(value);
    }
    return out;
}
export function createTaskGraphSimulator(options) {
    const workerCount = validateWorkerCount(options.workers);
    const tasks = new Map();
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
function validateWorkerCount(workers) {
    if (!Number.isInteger(workers) || workers <= 0) {
        throw new SimulatorError("workers must be a positive integer");
    }
    return workers;
}
function normalizeTask(task) {
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
function runTaskGraph(tasks, workerCount) {
    const taskById = new Map(tasks.map((task) => [task.id, task]));
    validateTaskDependencies(tasks, taskById);
    const events = [];
    const completed = new Set();
    const started = new Set();
    const announcedReady = new Set();
    const completedTaskIds = [];
    const workers = Array.from({ length: workerCount }, (_, worker) => ({
        worker,
        taskId: undefined,
        finishTime: Number.POSITIVE_INFINITY,
    }));
    let time = 0;
    const emit = (event) => {
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
            if (!next)
                break;
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
        if (startedThisTick)
            continue;
        const nextFinishTime = Math.min(...workers.map((worker) => worker.finishTime));
        if (!Number.isFinite(nextFinishTime)) {
            throw new SimulatorError("task graph has a dependency cycle");
        }
        time = nextFinishTime;
        for (const worker of workers
            .filter((item) => item.finishTime === time && item.taskId !== undefined)
            .sort((a, b) => a.worker - b.worker)) {
            const taskId = worker.taskId;
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
function validateTaskDependencies(tasks, taskById) {
    for (const task of tasks) {
        for (const dependency of task.dependsOn) {
            if (!taskById.has(dependency)) {
                throw new SimulatorError(`task ${task.id} depends on missing task: ${dependency}`);
            }
        }
    }
}
function readyTasks(tasks, completed, started) {
    return tasks
        .filter((task) => !completed.has(task.id) &&
        !started.has(task.id) &&
        task.dependsOn.every((dependency) => completed.has(dependency)))
        .sort((a, b) => a.id.localeCompare(b.id));
}
function cloneTaskGraphEvent(event) {
    return { ...event };
}
function uniqueSorted(values) {
    return [...new Set(values)].sort();
}
//# sourceMappingURL=index.js.map