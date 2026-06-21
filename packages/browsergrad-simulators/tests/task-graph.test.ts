import { describe, expect, it } from "vitest";
import { createTaskGraphSimulator, SimulatorError } from "../src/index";

describe("createTaskGraphSimulator", () => {
  it("runs dependency-constrained tasks with deterministic worker traces", () => {
    const simulator = createTaskGraphSimulator({ workers: 2 });
    simulator.addTask({ id: "load", duration: 2 });
    simulator.addTask({ id: "decode", duration: 3 });
    simulator.addTask({ id: "train", duration: 4, dependsOn: ["decode", "load"] });

    const result = simulator.run();

    expect(result.makespan).toBe(7);
    expect(result.completedTaskIds).toEqual(["load", "decode", "train"]);
    expect(result.events).toEqual([
      { time: 0, kind: "task-ready", taskId: "decode" },
      { time: 0, kind: "task-ready", taskId: "load" },
      { time: 0, kind: "task-start", taskId: "decode", worker: 0 },
      { time: 0, kind: "task-start", taskId: "load", worker: 1 },
      { time: 2, kind: "task-finish", taskId: "load", worker: 1 },
      { time: 3, kind: "task-finish", taskId: "decode", worker: 0 },
      { time: 3, kind: "task-ready", taskId: "train" },
      { time: 3, kind: "task-start", taskId: "train", worker: 0 },
      { time: 7, kind: "task-finish", taskId: "train", worker: 0 },
    ]);
  });

  it("returns immutable run snapshots", () => {
    const simulator = createTaskGraphSimulator({ workers: 1 });
    simulator.addTask({ id: "load", duration: 1 });

    const result = simulator.run();
    result.events[0]!.taskId = "mutated";
    result.completedTaskIds.push("mutated");

    expect(simulator.run().events[0]?.taskId).toBe("load");
    expect(simulator.run().completedTaskIds).toEqual(["load"]);
  });

  it("rejects duplicate tasks, missing deps, cycles, and invalid options", () => {
    expect(() => createTaskGraphSimulator({ workers: 0 })).toThrow(SimulatorError);

    const duplicate = createTaskGraphSimulator({ workers: 1 });
    duplicate.addTask({ id: "a", duration: 1 });
    expect(() => duplicate.addTask({ id: "a", duration: 1 })).toThrow(SimulatorError);
    expect(() => duplicate.addTask({ id: "b", duration: 0 })).toThrow(SimulatorError);

    const missing = createTaskGraphSimulator({ workers: 1 });
    missing.addTask({ id: "a", duration: 1, dependsOn: ["ghost"] });
    expect(() => missing.run()).toThrow(SimulatorError);

    const cycle = createTaskGraphSimulator({ workers: 2 });
    cycle.addTask({ id: "a", duration: 1, dependsOn: ["b"] });
    cycle.addTask({ id: "b", duration: 1, dependsOn: ["a"] });
    expect(() => cycle.run()).toThrow(SimulatorError);
  });
});
