import {
  createHostedScalingApiMock,
  fitPowerLawScalingLaw,
  ScalingApiError,
  selectExperimentsForDispatch,
  type BudgetSummary,
  type DashboardResponse,
  type ExperimentResponse,
  type FinalSubmissionResponse,
  type HostedScalingApiMock,
  type HostedScalingApiMockOptions,
  type JsonObject,
  type PowerLawFitOptions,
  type PowerLawScalingFit,
  type SubmitResponse,
} from "@unlocalhosted/browsergrad-scaling";

export {
  fitPowerLawScalingLaw,
  ScalingApiError,
  selectExperimentsForDispatch,
  type JsonObject,
  type PowerLawFitOptions,
  type PowerLawScalingFit,
};

export interface HostedTrainingUser {
  readonly userId: string;
  readonly apiKey: string;
}

export interface HostedTrainingApiFixtureOptions
  extends Omit<HostedScalingApiMockOptions, "users"> {
  readonly users?: readonly HostedTrainingUser[];
}

export interface TrainingBudgetSummary {
  readonly usedSeconds: number;
  readonly remainingSeconds: number;
  readonly totalBudgetSeconds: number;
}

export interface TrainingSubmitResponse {
  readonly experimentId: number;
  readonly budgetSummary: TrainingBudgetSummary;
}

export interface TrainingExperimentResponse {
  readonly experimentId: number;
  readonly userId: string;
  readonly trainingConfig: JsonObject;
  readonly trainingConfigUniqueId: string;
  readonly status: ExperimentResponse["status"];
}

export interface TrainingFinalSubmissionResponse {
  readonly trainingConfig: JsonObject;
  readonly predictedFinalLoss: number;
  readonly submittedAt: string;
}

export interface HostedTrainingApiFixture {
  registerUser(user: HostedTrainingUser): void;
  dashboard(): DashboardResponse;
  getBudget(apiKey: string): TrainingBudgetSummary;
  submitExperiment(
    apiKey: string,
    trainingConfig: JsonObject,
  ): TrainingSubmitResponse;
  listExperiments(apiKey: string): TrainingExperimentResponse[];
  getExperiment(apiKey: string, experimentId: number): TrainingExperimentResponse;
  saveFinalSubmission(
    apiKey: string,
    trainingConfig: JsonObject,
    predictedFinalLoss: number,
  ): TrainingFinalSubmissionResponse;
  getFinalSubmission(apiKey: string): TrainingFinalSubmissionResponse | null;
}

export function createHostedTrainingApiFixture(
  options: HostedTrainingApiFixtureOptions = {},
): HostedTrainingApiFixture {
  const { users: fixtureUsers, ...scalingOptions } = options;
  const users = fixtureUsers?.map((user) => ({
    sunetId: user.userId,
    apiKey: user.apiKey,
  }));
  const apiOptions: HostedScalingApiMockOptions =
    users === undefined ? scalingOptions : { ...scalingOptions, users };
  const api = createHostedScalingApiMock(apiOptions);
  return createHostedTrainingApiFixtureAdapter(api);
}

function createHostedTrainingApiFixtureAdapter(
  api: HostedScalingApiMock,
): HostedTrainingApiFixture {
  return {
    registerUser(user) {
      api.registerUser({ sunetId: user.userId, apiKey: user.apiKey });
    },
    dashboard: api.dashboard,
    getBudget(apiKey) {
      return budgetSummary(api.getBudget(apiKey));
    },
    submitExperiment(apiKey, trainingConfig) {
      return submitResponse(api.submitExperiment(apiKey, trainingConfig));
    },
    listExperiments(apiKey) {
      return api.listExperiments(apiKey).map(experimentResponse);
    },
    getExperiment(apiKey, experimentId) {
      return experimentResponse(api.getExperiment(apiKey, experimentId));
    },
    saveFinalSubmission(apiKey, trainingConfig, predictedFinalLoss) {
      return finalSubmissionResponse(
        api.saveFinalSubmission(apiKey, trainingConfig, predictedFinalLoss),
      );
    },
    getFinalSubmission(apiKey) {
      const submission = api.getFinalSubmission(apiKey);
      return submission ? finalSubmissionResponse(submission) : null;
    },
  };
}

function budgetSummary(summary: BudgetSummary): TrainingBudgetSummary {
  return {
    usedSeconds: summary.used_seconds,
    remainingSeconds: summary.remaining_seconds,
    totalBudgetSeconds: summary.total_budget_seconds,
  };
}

function submitResponse(response: SubmitResponse): TrainingSubmitResponse {
  return {
    experimentId: response.experiment_id,
    budgetSummary: budgetSummary(response.budget_summary),
  };
}

function experimentResponse(
  response: ExperimentResponse,
): TrainingExperimentResponse {
  return {
    experimentId: response.experiment_id,
    userId: response.user_sunet_id,
    trainingConfig: response.training_config,
    trainingConfigUniqueId: response.training_config_unique_id,
    status: response.status,
  };
}

function finalSubmissionResponse(
  response: FinalSubmissionResponse,
): TrainingFinalSubmissionResponse {
  return {
    trainingConfig: response.training_config,
    predictedFinalLoss: response.predicted_final_loss,
    submittedAt: response.submitted_at,
  };
}
