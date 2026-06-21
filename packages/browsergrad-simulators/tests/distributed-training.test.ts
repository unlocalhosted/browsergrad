import { describe, expect, it } from "vitest";
import {
  SimulatorError,
  simulateFsdpGradientReduceScatter,
  simulateFsdpParameterSharding,
  simulateDdpGradientSynchronization,
  simulateShardedAdamWStep,
} from "../src/index";

describe("simulateDdpGradientSynchronization", () => {
  it("averages trainable parameter gradients and replicates them to every rank", () => {
    const result = simulateDdpGradientSynchronization({
      parameters: [
        { name: "fc1.weight", trainable: true },
        { name: "fc2.bias", trainable: false },
        { name: "fc3.weight", trainable: true },
      ],
      rankGradients: [
        {
          "fc1.weight": [1, 3],
          "fc3.weight": [5, 7],
        },
        {
          "fc1.weight": [3, 5],
          "fc3.weight": [1, 9],
        },
      ],
    });

    expect(result.synchronizedGradients).toEqual([
      {
        "fc1.weight": [2, 4],
        "fc3.weight": [3, 8],
      },
      {
        "fc1.weight": [2, 4],
        "fc3.weight": [3, 8],
      },
    ]);
    expect(result.events).toEqual([
      {
        step: 0,
        kind: "ddp-gradient-all-reduce",
        parameter: "fc1.weight",
        participants: [0, 1],
        inputGradients: [
          [1, 3],
          [3, 5],
        ],
        outputGradient: [2, 4],
      },
      {
        step: 1,
        kind: "ddp-gradient-all-reduce",
        parameter: "fc3.weight",
        participants: [0, 1],
        inputGradients: [
          [5, 7],
          [1, 9],
        ],
        outputGradient: [3, 8],
      },
    ]);
  });

  it("rejects missing or mismatched local gradients", () => {
    expect(() =>
      simulateDdpGradientSynchronization({
        parameters: [{ name: "fc.weight" }],
        rankGradients: [{ "fc.weight": [1, 2] }, { "fc.weight": [1] }],
      }),
    ).toThrow(SimulatorError);
  });
});

describe("simulateShardedAdamWStep", () => {
  it("updates full parameters while assigning optimizer state to rank-local owners", () => {
    const result = simulateShardedAdamWStep({
      ranks: 2,
      parameters: [
        {
          name: "fc1.weight",
          values: [1, 2],
          gradients: [0.5, -0.25],
        },
        {
          name: "fc2.weight",
          values: [3],
          gradients: [0.25],
        },
      ],
      optimizer: {
        lr: 0.1,
        weightDecay: 0.01,
        betas: [0.9, 0.999],
        eps: 0,
      },
    });

    expect(result.updatedParameters).toEqual({
      "fc1.weight": [expect.closeTo(0.899, 12), expect.closeTo(2.098, 12)],
      "fc2.weight": [expect.closeTo(2.897, 12)],
    });
    expect(result.ownership).toEqual([
      { rank: 0, parameters: ["fc1.weight"], elements: 2 },
      { rank: 1, parameters: ["fc2.weight"], elements: 1 },
    ]);
    expect(result.nextState["fc1.weight"]).toEqual({
      step: 1,
      expAvg: [0.04999999999999999, -0.024999999999999994],
      expAvgSq: [0.0002500000000000002, 0.00006250000000000005],
    });
    expect(result.events.map((event) => event.kind)).toEqual([
      "optimizer-state-shard",
      "optimizer-state-shard",
      "sharded-adamw-step",
    ]);
  });
});

describe("FSDP sharding simulators", () => {
  it("shards trainable parameters, replicates explicit buffers, and all-gathers full values", () => {
    const result = simulateFsdpParameterSharding({
      ranks: 2,
      parameters: [
        { name: "embedding.weight", values: [0, 1, 2, 3, 4], sharded: true },
        { name: "norm.weight", values: [10, 11], sharded: false },
      ],
    });

    expect(result.rankShards).toEqual([
      {
        "embedding.weight": { start: 0, end: 3, values: [0, 1, 2] },
        "norm.weight": { start: 0, end: 2, values: [10, 11], replicated: true },
      },
      {
        "embedding.weight": { start: 3, end: 5, values: [3, 4] },
        "norm.weight": { start: 0, end: 2, values: [10, 11], replicated: true },
      },
    ]);
    expect(result.fullParameters).toEqual({
      "embedding.weight": [0, 1, 2, 3, 4],
      "norm.weight": [10, 11],
    });
    expect(result.events.map((event) => event.kind)).toEqual([
      "fsdp-shard",
      "fsdp-replicate",
      "fsdp-all-gather",
    ]);
  });

  it("reduce-scatters averaged gradients back to owned shards", () => {
    const sharding = simulateFsdpParameterSharding({
      ranks: 2,
      parameters: [
        { name: "embedding.weight", values: [0, 1, 2, 3, 4], sharded: true },
        { name: "norm.weight", values: [10, 11], sharded: false },
      ],
    });

    const result = simulateFsdpGradientReduceScatter({
      shardPlan: sharding.shardPlan,
      rankGradients: [
        {
          "embedding.weight": [1, 3, 5, 7, 9],
          "norm.weight": [1, 1],
        },
        {
          "embedding.weight": [3, 5, 7, 9, 11],
          "norm.weight": [3, 3],
        },
      ],
    });

    expect(result.rankGradientShards).toEqual([
      {
        "embedding.weight": { start: 0, end: 3, values: [2, 4, 6] },
        "norm.weight": { start: 0, end: 2, values: [2, 2], replicated: true },
      },
      {
        "embedding.weight": { start: 3, end: 5, values: [8, 10] },
        "norm.weight": { start: 0, end: 2, values: [2, 2], replicated: true },
      },
    ]);
    expect(result.events.map((event) => event.kind)).toEqual([
      "fsdp-reduce-scatter",
      "fsdp-replicated-gradient-all-reduce",
    ]);
  });
});
