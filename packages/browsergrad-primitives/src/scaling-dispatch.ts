import { ScalingApiError } from "./scaling-error.js";
import type {
  DispatchExperiment,
  DispatchSelection,
  DispatchSelectorOptions,
} from "./scaling-types.js";

export function selectExperimentsForDispatch<T extends DispatchExperiment>(
  experiments: readonly T[],
  options: DispatchSelectorOptions,
): DispatchSelection<T> {
  if (
    !Number.isInteger(options.maxConcurrentWorkers) ||
    options.maxConcurrentWorkers < 0
  ) {
    throw new ScalingApiError(400, "maxConcurrentWorkers must be a non-negative integer");
  }

  const runningCounts = new Map<string, number>();
  let currentlyRunningJobs = 0;
  for (const experiment of experiments) {
    if (experiment.status.status_type !== "running") continue;
    currentlyRunningJobs += 1;
    runningCounts.set(
      experiment.userId,
      (runningCounts.get(experiment.userId) ?? 0) + 1,
    );
  }

  const capacity = Math.max(options.maxConcurrentWorkers - currentlyRunningJobs, 0);
  if (capacity === 0) {
    return {
      experiments: [],
      currentlyRunningJobs,
      zeroRunningUserExperiments: 0,
    };
  }

  const queuedByUser = new Map<string, T[]>();
  for (const experiment of experiments) {
    if (experiment.status.status_type !== "queued") continue;
    const queuedAt = experiment.status.queued_at;
    if (!queuedAt) {
      throw new ScalingApiError(400, "queued experiments require status.queued_at");
    }
    const existing = queuedByUser.get(experiment.userId) ?? [];
    existing.push(experiment);
    queuedByUser.set(experiment.userId, existing);
  }

  type Ranked = {
    readonly experiment: T;
    readonly effectiveRunningCount: number;
    readonly queuedAtMs: number;
  };
  const ranked: Ranked[] = [];
  for (const [user, queued] of queuedByUser) {
    const sorted = [...queued].sort(compareQueuedExperiments);
    const initialRunningCount = runningCounts.get(user) ?? 0;
    for (let index = 0; index < sorted.length; index++) {
      const experiment = sorted[index];
      if (!experiment) continue;
      ranked.push({
        experiment,
        effectiveRunningCount: initialRunningCount + index,
        queuedAtMs: queuedAtMs(experiment),
      });
    }
  }

  const selected = ranked
    .sort((left, right) => {
      const runningDelta =
        left.effectiveRunningCount - right.effectiveRunningCount;
      if (runningDelta !== 0) return runningDelta;
      const timeDelta = left.queuedAtMs - right.queuedAtMs;
      if (timeDelta !== 0) return timeDelta;
      return compareExperimentId(left.experiment.id, right.experiment.id);
    })
    .slice(0, capacity);

  return {
    experiments: selected.map((item) => item.experiment),
    currentlyRunningJobs,
    zeroRunningUserExperiments: selected.filter(
      (item) => item.effectiveRunningCount === 0,
    ).length,
  };
}

function compareQueuedExperiments<T extends DispatchExperiment>(
  left: T,
  right: T,
): number {
  const timeDelta = queuedAtMs(left) - queuedAtMs(right);
  if (timeDelta !== 0) return timeDelta;
  return compareExperimentId(left.id, right.id);
}

function queuedAtMs(experiment: DispatchExperiment): number {
  const value = experiment.status.queued_at;
  if (!value) {
    throw new ScalingApiError(400, "queued experiments require status.queued_at");
  }
  const ms = Date.parse(value);
  if (!Number.isFinite(ms)) {
    throw new ScalingApiError(400, "status.queued_at must be an ISO timestamp");
  }
  return ms;
}

function compareExperimentId(left: number | string, right: number | string): number {
  if (typeof left === "number" && typeof right === "number") {
    return left - right;
  }
  return String(left).localeCompare(String(right));
}
