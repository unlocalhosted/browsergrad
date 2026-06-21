import { describe, expect, it } from "vitest";
import {
  createDeterministicMesh,
  createTaskGraphSimulator,
  partitionStaticWork,
  SimulatorError,
  simulateCs149ArraySumVector,
  simulateCs149ClampedExpVector,
  simulateDdpGradientSynchronization,
  simulateFsdpGradientReduceScatter,
  simulateFsdpParameterSharding,
  simulateShardedAdamWStep,
  type DeterministicMesh,
  type DdpGradientSynchronizationResult,
  type FsdpGradientReduceScatterResult,
  type FsdpParameterShardingResult,
  type SimulationEvent,
  type SimdKernelStats,
  type ShardedAdamWResult,
  type StaticWorkPartition,
  type TaskGraphEvent,
  type TaskGraphSimulator,
} from "../src/index";

describe("public surface", () => {
  it("exports deterministic mesh primitives", () => {
    expect(typeof createDeterministicMesh).toBe("function");
    expect(typeof createTaskGraphSimulator).toBe("function");
    expect(typeof partitionStaticWork).toBe("function");
    expect(typeof simulateCs149ArraySumVector).toBe("function");
    expect(typeof simulateCs149ClampedExpVector).toBe("function");
    expect(typeof simulateDdpGradientSynchronization).toBe("function");
    expect(typeof simulateFsdpParameterSharding).toBe("function");
    expect(typeof simulateFsdpGradientReduceScatter).toBe("function");
    expect(typeof simulateShardedAdamWStep).toBe("function");
    expect(typeof SimulatorError).toBe("function");
  });

  it("types mesh and events for compile-time consumers", () => {
    const mesh: DeterministicMesh = createDeterministicMesh({ ranks: 2 });
    const event: SimulationEvent = {
      step: 0,
      kind: "barrier",
      tag: "sync",
      participants: [0, 1],
    };
    expect(mesh.rankCount).toBe(2);
    expect(event.participants).toEqual([0, 1]);
  });

  it("types task graph events for compile-time consumers", () => {
    const simulator: TaskGraphSimulator = createTaskGraphSimulator({ workers: 1 });
    const event: TaskGraphEvent = {
      time: 0,
      kind: "task-start",
      taskId: "load",
      worker: 0,
    };
    expect(simulator.workerCount).toBe(1);
    expect(event.taskId).toBe("load");
  });

  it("types CS149 CPU/SIMD simulator results for compile-time consumers", () => {
    const clamped = simulateCs149ClampedExpVector({
      values: [2],
      exponents: [1],
      vectorWidth: 4,
    });
    const stats: SimdKernelStats = clamped.stats;
    const partitions: StaticWorkPartition[] = partitionStaticWork({
      items: 2,
      workers: 1,
    });

    expect(clamped.output).toEqual([2]);
    expect(stats.vectorInstructions).toBeGreaterThan(0);
    expect(partitions[0]?.ranges).toEqual([{ start: 0, end: 2 }]);
  });

  it("types distributed training simulator results for compile-time consumers", () => {
    const ddp: DdpGradientSynchronizationResult = simulateDdpGradientSynchronization({
      parameters: [{ name: "w" }],
      rankGradients: [{ w: [1] }, { w: [3] }],
    });
    const fsdp: FsdpParameterShardingResult = simulateFsdpParameterSharding({
      ranks: 2,
      parameters: [{ name: "w", values: [1, 2] }],
    });
    const scattered: FsdpGradientReduceScatterResult =
      simulateFsdpGradientReduceScatter({
        shardPlan: fsdp.shardPlan,
        rankGradients: [{ w: [1, 3] }, { w: [3, 5] }],
      });
    const optimizer: ShardedAdamWResult = simulateShardedAdamWStep({
      ranks: 2,
      parameters: [{ name: "w", values: [1], gradients: [1] }],
      optimizer: { lr: 0.1 },
    });

    expect(ddp.synchronizedGradients[0]?.w).toEqual([2]);
    expect(scattered.rankGradientShards[0]?.w?.values).toEqual([2]);
    expect(optimizer.ownership[0]?.parameters).toEqual(["w"]);
  });
});
