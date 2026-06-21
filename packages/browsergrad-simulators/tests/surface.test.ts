import { describe, expect, it } from "vitest";
import {
  createDeterministicMesh,
  createTaskGraphSimulator,
  SimulatorError,
  type DeterministicMesh,
  type SimulationEvent,
  type TaskGraphEvent,
  type TaskGraphSimulator,
} from "../src/index";

describe("public surface", () => {
  it("exports deterministic mesh primitives", () => {
    expect(typeof createDeterministicMesh).toBe("function");
    expect(typeof createTaskGraphSimulator).toBe("function");
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
});
