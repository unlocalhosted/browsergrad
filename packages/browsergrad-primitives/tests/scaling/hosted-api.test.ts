import { describe, expect, it } from "vitest";
import {
  ScalingApiError,
  createHostedTrainingApiFixture,
} from "../../src/scaling";

function trainingConfig(hiddenSize: number, maxRuntimeSeconds: number) {
  return {
    architecture_config: {
      hidden_size: hiddenSize,
      num_hidden_layers: 1,
    },
    optimizer_config: {
      learning_rate: 0.001,
    },
    train_batch_size: 2,
    total_train_tokens: 2048,
    max_runtime_seconds: maxRuntimeSeconds,
  };
}

describe("createHostedTrainingApiFixture", () => {
  it("matches hosted-training budget, submit, duplicate, list, and final-submission behavior", () => {
    const api = createHostedTrainingApiFixture({
      totalBudgetSeconds: 60,
      now: () => new Date("2026-01-01T00:00:00.000Z"),
    });
    api.registerUser({ userId: "alice", apiKey: "test-api-key" });

    expect(api.dashboard()).toEqual({
      status: 200,
      contentType: "text/html; charset=utf-8",
      body: expect.stringContaining("Scaling Experiments Dashboard"),
    });
    expect(api.getBudget("test-api-key")).toEqual({
      usedSeconds: 0,
      remainingSeconds: 60,
      totalBudgetSeconds: 60,
    });

    const first = api.submitExperiment("test-api-key", trainingConfig(202, 10));
    expect(first).toEqual({
      experimentId: 1,
      budgetSummary: {
        usedSeconds: 10,
        remainingSeconds: 50,
        totalBudgetSeconds: 60,
      },
    });
    const second = api.submitExperiment("test-api-key", trainingConfig(204, 20));
    expect(second.experimentId).toBe(2);

    expect(() => {
      api.submitExperiment("test-api-key", trainingConfig(202, 10));
    }).toThrow(ScalingApiError);
    try {
      api.submitExperiment("test-api-key", trainingConfig(202, 10));
    } catch (error) {
      expect(error).toMatchObject({
        status: 409,
        detail: "experiment already exists for this training config",
      });
    }

    const experiments = api.listExperiments("test-api-key");
    expect(experiments.map((item) => item.experimentId)).toEqual([1, 2]);
    expect(experiments.map((item) => item.status.status_type)).toEqual([
      "queued",
      "queued",
    ]);
    expect(api.getExperiment("test-api-key", 1)).toEqual(experiments[0]);

    const final = api.saveFinalSubmission(
      "test-api-key",
      trainingConfig(256, 15),
      2.75,
    );
    expect(final).toEqual({
      trainingConfig: trainingConfig(256, 15),
      predictedFinalLoss: 2.75,
      submittedAt: "2026-01-01T00:00:00.000Z",
    });
    expect(api.getFinalSubmission("test-api-key")).toEqual(final);
  });
});
