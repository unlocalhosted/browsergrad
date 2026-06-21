export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];
export interface JsonObject {
  readonly [key: string]: JsonValue;
}

export interface BudgetSummary {
  readonly used_seconds: number;
  readonly remaining_seconds: number;
  readonly total_budget_seconds: number;
}

export interface SubmitResponse {
  readonly experiment_id: number;
  readonly budget_summary: BudgetSummary;
}

export interface QueuedExperimentStatus {
  readonly status_type: "queued";
  readonly queued_at: string;
}

export interface RunningExperimentStatus {
  readonly status_type: "running";
  readonly queued_at: string;
  readonly dispatched_at: string;
  readonly run_id: string;
}

export interface CompletedExperimentStatus {
  readonly status_type: "completed";
  readonly queued_at: string;
  readonly dispatched_at?: string;
  readonly completed_at: string;
  readonly used_runtime_seconds: number;
}

export interface FailedExperimentStatus {
  readonly status_type: "failed";
  readonly queued_at: string;
  readonly dispatched_at?: string;
  readonly failed_at: string;
  readonly used_runtime_seconds: number;
  readonly detail?: string;
}

export type ExperimentStatus =
  | QueuedExperimentStatus
  | RunningExperimentStatus
  | CompletedExperimentStatus
  | FailedExperimentStatus;

export interface ExperimentResponse {
  readonly experiment_id: number;
  readonly user_sunet_id: string;
  readonly training_config: JsonObject;
  readonly training_config_unique_id: string;
  readonly status: ExperimentStatus;
}

export interface FinalSubmissionResponse {
  readonly training_config: JsonObject;
  readonly predicted_final_loss: number;
  readonly submitted_at: string;
}

export interface DashboardResponse {
  readonly status: 200;
  readonly contentType: string;
  readonly body: string;
}

export interface HostedScalingApiUser {
  readonly sunetId: string;
  readonly apiKey: string;
}

export interface HostedScalingApiMockOptions {
  readonly totalBudgetSeconds?: number;
  readonly now?: () => Date;
  readonly users?: readonly HostedScalingApiUser[];
}

export interface HostedScalingApiMock {
  registerUser(user: HostedScalingApiUser): void;
  dashboard(): DashboardResponse;
  getBudget(apiKey: string): BudgetSummary;
  submitExperiment(apiKey: string, trainingConfig: JsonObject): SubmitResponse;
  listExperiments(apiKey: string): ExperimentResponse[];
  getExperiment(apiKey: string, experimentId: number): ExperimentResponse;
  saveFinalSubmission(
    apiKey: string,
    trainingConfig: JsonObject,
    predictedFinalLoss: number,
  ): FinalSubmissionResponse;
  getFinalSubmission(apiKey: string): FinalSubmissionResponse | null;
}

export interface DispatchExperiment {
  readonly id: number | string;
  readonly user_sunet_id: string;
  readonly training_config: JsonObject;
  readonly status: {
    readonly status_type: string;
    readonly queued_at?: string;
  };
}

export interface DispatchSelection<T extends DispatchExperiment> {
  readonly experiments: readonly T[];
  readonly currentlyRunningJobs: number;
  readonly zeroRunningUserExperiments: number;
}

export interface DispatchSelectorOptions {
  readonly maxConcurrentWorkers: number;
}

export interface PowerLawScalingFit {
  readonly x: string;
  readonly y: string;
  readonly slope: number;
  readonly intercept: number;
  readonly exponent: number;
  readonly multiplier: number;
  readonly rSquared: number;
  predict(x: number): number;
}

export interface PowerLawFitOptions {
  readonly x: string;
  readonly y: string;
}

export class ScalingApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ScalingApiError";
    this.status = status;
    this.detail = detail;
  }
}

interface UserRecord {
  readonly sunetId: string;
  readonly apiKey: string;
}

interface ExperimentRecord {
  readonly experimentId: number;
  readonly userSunetId: string;
  readonly trainingConfig: JsonObject;
  readonly trainingConfigUniqueId: string;
  status: ExperimentStatus;
}

export function createHostedScalingApiMock(
  options: HostedScalingApiMockOptions = {},
): HostedScalingApiMock {
  const totalBudgetSeconds = options.totalBudgetSeconds ?? 3_600;
  if (!Number.isFinite(totalBudgetSeconds) || totalBudgetSeconds < 0) {
    throw new ScalingApiError(500, "totalBudgetSeconds must be non-negative");
  }

  const now = options.now ?? (() => new Date());
  const usersByApiKey = new Map<string, UserRecord>();
  const experiments: ExperimentRecord[] = [];
  const finalSubmissions = new Map<string, FinalSubmissionResponse>();

  const registerUser = (user: HostedScalingApiUser): void => {
    if (user.sunetId.length === 0 || user.apiKey.length === 0) {
      throw new ScalingApiError(400, "sunetId and apiKey are required");
    }
    usersByApiKey.set(user.apiKey, {
      sunetId: user.sunetId,
      apiKey: user.apiKey,
    });
  };

  for (const user of options.users ?? []) {
    registerUser(user);
  }

  const requireUser = (apiKey: string): UserRecord => {
    const user = usersByApiKey.get(apiKey);
    if (!user) {
      throw new ScalingApiError(401, "invalid API key");
    }
    return user;
  };

  const budgetForUser = (sunetId: string): BudgetSummary => {
    const used = experiments
      .filter((experiment) => experiment.userSunetId === sunetId)
      .reduce((sum, experiment) => sum + budgetSecondsForExperiment(experiment), 0);
    return {
      used_seconds: used,
      remaining_seconds: totalBudgetSeconds - used,
      total_budget_seconds: totalBudgetSeconds,
    };
  };

  return {
    registerUser,
    dashboard() {
      return {
        status: 200,
        contentType: "text/html; charset=utf-8",
        body: "<!doctype html><title>Scaling Experiments Dashboard</title>",
      };
    },
    getBudget(apiKey) {
      return budgetForUser(requireUser(apiKey).sunetId);
    },
    submitExperiment(apiKey, trainingConfig) {
      const user = requireUser(apiKey);
      const config = cloneJsonObject(trainingConfig);
      const uniqueId = stableTrainingConfigId(config);
      if (
        experiments.some(
          (experiment) =>
            experiment.userSunetId === user.sunetId &&
            experiment.trainingConfigUniqueId === uniqueId,
        )
      ) {
        throw new ScalingApiError(
          409,
          "experiment already exists for this training config",
        );
      }
      const projectedBudget =
        budgetForUser(user.sunetId).used_seconds + maxRuntimeSeconds(config);
      if (projectedBudget > totalBudgetSeconds) {
        throw new ScalingApiError(402, "training budget exceeded");
      }
      const experimentId = experiments.length + 1;
      experiments.push({
        experimentId,
        userSunetId: user.sunetId,
        trainingConfig: config,
        trainingConfigUniqueId: uniqueId,
        status: {
          status_type: "queued",
          queued_at: now().toISOString(),
        },
      });
      return {
        experiment_id: experimentId,
        budget_summary: budgetForUser(user.sunetId),
      };
    },
    listExperiments(apiKey) {
      const user = requireUser(apiKey);
      return experiments
        .filter((experiment) => experiment.userSunetId === user.sunetId)
        .map(experimentResponse);
    },
    getExperiment(apiKey, experimentId) {
      const user = requireUser(apiKey);
      const experiment = experiments.find(
        (item) =>
          item.userSunetId === user.sunetId &&
          item.experimentId === experimentId,
      );
      if (!experiment) {
        throw new ScalingApiError(404, "experiment not found");
      }
      return experimentResponse(experiment);
    },
    saveFinalSubmission(apiKey, trainingConfig, predictedFinalLoss) {
      const user = requireUser(apiKey);
      if (!Number.isFinite(predictedFinalLoss)) {
        throw new ScalingApiError(400, "predicted_final_loss must be finite");
      }
      const submission = {
        training_config: cloneJsonObject(trainingConfig),
        predicted_final_loss: predictedFinalLoss,
        submitted_at: now().toISOString(),
      };
      finalSubmissions.set(user.sunetId, submission);
      return cloneFinalSubmission(submission);
    },
    getFinalSubmission(apiKey) {
      const user = requireUser(apiKey);
      const submission = finalSubmissions.get(user.sunetId);
      return submission ? cloneFinalSubmission(submission) : null;
    },
  };
}

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
      experiment.user_sunet_id,
      (runningCounts.get(experiment.user_sunet_id) ?? 0) + 1,
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
    const existing = queuedByUser.get(experiment.user_sunet_id) ?? [];
    existing.push(experiment);
    queuedByUser.set(experiment.user_sunet_id, existing);
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

export function fitPowerLawScalingLaw<T extends Record<string, number>>(
  samples: readonly T[],
  options: PowerLawFitOptions,
): PowerLawScalingFit {
  if (samples.length < 2) {
    throw new ScalingApiError(400, "at least two samples are required");
  }

  const points = samples.map((sample, index) => {
    const x = positiveSampleValue(sample, options.x, index);
    const y = positiveSampleValue(sample, options.y, index);
    return {
      logX: Math.log(x),
      logY: Math.log(y),
    };
  });
  const meanX = mean(points.map((point) => point.logX));
  const meanY = mean(points.map((point) => point.logY));
  let covariance = 0;
  let varianceX = 0;
  for (const point of points) {
    const dx = point.logX - meanX;
    covariance += dx * (point.logY - meanY);
    varianceX += dx * dx;
  }
  if (varianceX === 0) {
    throw new ScalingApiError(400, `${options.x} values must not all match`);
  }
  const slope = covariance / varianceX;
  const intercept = meanY - slope * meanX;
  const predictions = points.map((point) => intercept + slope * point.logX);
  const residualSumSquares = points.reduce((sum, point, index) => {
    const prediction = predictions[index];
    if (prediction === undefined) return sum;
    const residual = point.logY - prediction;
    return sum + residual * residual;
  }, 0);
  const totalSumSquares = points.reduce((sum, point) => {
    const centered = point.logY - meanY;
    return sum + centered * centered;
  }, 0);
  const rSquared =
    totalSumSquares === 0 ? 1 : 1 - residualSumSquares / totalSumSquares;
  const multiplier = Math.exp(intercept);

  return {
    x: options.x,
    y: options.y,
    slope,
    intercept,
    exponent: slope,
    multiplier,
    rSquared,
    predict(x) {
      if (!Number.isFinite(x) || x <= 0) {
        throw new ScalingApiError(400, `${options.x} must be positive`);
      }
      return multiplier * x ** slope;
    },
  };
}

function experimentResponse(experiment: ExperimentRecord): ExperimentResponse {
  return {
    experiment_id: experiment.experimentId,
    user_sunet_id: experiment.userSunetId,
    training_config: cloneJsonObject(experiment.trainingConfig),
    training_config_unique_id: experiment.trainingConfigUniqueId,
    status: cloneStatus(experiment.status),
  };
}

function budgetSecondsForExperiment(experiment: ExperimentRecord): number {
  const status = experiment.status;
  if (status.status_type === "completed") {
    return Math.min(
      Math.max(1, status.used_runtime_seconds),
      maxRuntimeSeconds(experiment.trainingConfig),
    );
  }
  if (status.status_type === "failed") {
    return Math.min(
      Math.max(1, status.used_runtime_seconds),
      maxRuntimeSeconds(experiment.trainingConfig),
    );
  }
  if (status.status_type === "queued" || status.status_type === "running") {
    return maxRuntimeSeconds(experiment.trainingConfig);
  }
  return 0;
}

function maxRuntimeSeconds(config: JsonObject): number {
  const raw = config.max_runtime_seconds ?? config.maxRuntimeSeconds;
  if (typeof raw !== "number" || !Number.isFinite(raw) || raw < 0) {
    throw new ScalingApiError(
      400,
      "training_config.max_runtime_seconds must be a non-negative number",
    );
  }
  return raw;
}

function stableTrainingConfigId(config: JsonObject): string {
  return canonicalJson(config);
}

function cloneFinalSubmission(
  submission: FinalSubmissionResponse,
): FinalSubmissionResponse {
  return {
    training_config: cloneJsonObject(submission.training_config),
    predicted_final_loss: submission.predicted_final_loss,
    submitted_at: submission.submitted_at,
  };
}

function cloneStatus(status: ExperimentStatus): ExperimentStatus {
  return cloneJsonObject(status as unknown as JsonObject) as unknown as ExperimentStatus;
}

function cloneJsonObject(value: JsonObject): JsonObject {
  return JSON.parse(JSON.stringify(value)) as JsonObject;
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

function positiveSampleValue<T extends Record<string, number>>(
  sample: T,
  field: string,
  index: number,
): number {
  const value = sample[field];
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new ScalingApiError(400, `samples[${index}].${field} must be positive`);
  }
  return value;
}

function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function canonicalJson(value: JsonValue): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  return `{${Object.keys(value)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key] ?? null)}`)
    .join(",")}}`;
}
