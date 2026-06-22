import { describe, expect, it } from "vitest";
import { selectExperimentsForDispatch } from "../../src/scaling";

const NOW = "2026-01-01T00:00:00.000Z";

interface FixtureExperiment {
  readonly id: number;
  readonly userId: string;
  readonly training_config: { readonly model_seed: number };
  readonly status:
    | {
        readonly status_type: "queued";
        readonly queued_at: string;
      }
    | {
        readonly status_type: "running";
        readonly queued_at: string;
        readonly dispatched_at: string;
        readonly run_id: string;
      };
}

function minutesAfter(minutes: number): string {
  return new Date(Date.parse(NOW) + minutes * 60_000).toISOString();
}

function queued(
  id: number,
  user: string,
  modelSeed: number,
  queuedAtMinutes: number,
): FixtureExperiment {
  return {
    id,
    userId: user,
    training_config: { model_seed: modelSeed },
    status: {
      status_type: "queued",
      queued_at: minutesAfter(queuedAtMinutes),
    },
  };
}

function running(id: number, user: string, modelSeed: number): FixtureExperiment {
  return {
    id,
    userId: user,
    training_config: { model_seed: modelSeed },
    status: {
      status_type: "running",
      queued_at: NOW,
      dispatched_at: NOW,
      run_id: `${user}-${modelSeed}`,
    },
  };
}

describe("selectExperimentsForDispatch", () => {
  it("matches hosted-training dispatch fairness: running-count first, queue-time second", () => {
    const experiments: FixtureExperiment[] = [
      running(1, "user2", 1),
      running(2, "user2", 2),
      running(3, "user3", 3),
      running(4, "user3", 4),
      queued(5, "user1", 5, 10),
      queued(6, "user2", 6, 19),
      queued(7, "user3", 7, 9),
      queued(8, "user1", 8, 10),
      queued(9, "user2", 9, 19),
      queued(10, "user3", 10, 9),
      queued(11, "user1", 11, 10),
    ];

    const selection = selectExperimentsForDispatch(experiments, {
      maxConcurrentWorkers: 9,
    });

    expect(selection.currentlyRunningJobs).toBe(4);
    expect(selection.zeroRunningUserExperiments).toBe(1);
    expect(
      selection.experiments.map((experiment) => [
        experiment.training_config.model_seed,
        experiment.userId,
      ]),
    ).toEqual([
      [5, "user1"],
      [8, "user1"],
      [7, "user3"],
      [11, "user1"],
      [6, "user2"],
    ]);
  });

  it("reports running jobs when no worker capacity remains", () => {
    const selection = selectExperimentsForDispatch(
      [running(1, "user1", 1), queued(2, "user1", 2, 1)],
      { maxConcurrentWorkers: 1 },
    );

    expect(selection.experiments).toEqual([]);
    expect(selection.currentlyRunningJobs).toBe(1);
    expect(selection.zeroRunningUserExperiments).toBe(0);
  });
});
