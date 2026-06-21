import { describe, expect, it } from "vitest";
import { createDeterministicMesh, SimulatorError } from "../src/index";

describe("createDeterministicMesh", () => {
  it("records deterministic barrier, send, broadcast, and all-reduce traces", () => {
    const mesh = createDeterministicMesh({ ranks: 3 });

    mesh.barrier("start");
    mesh.send({ from: 2, to: 0, tag: "ready", payload: { shard: 2 } });
    mesh.broadcast({ from: 1, tag: "weights", payload: [4, 5] });
    const reduced = mesh.allReduce({
      tag: "grad-bucket-0",
      values: [1, 2, 3],
      op: "sum",
    });

    expect(reduced).toEqual([6, 6, 6]);
    expect(mesh.trace()).toEqual([
      { step: 0, kind: "barrier", tag: "start", participants: [0, 1, 2] },
      { step: 1, kind: "send", tag: "ready", from: 2, to: 0, payload: { shard: 2 } },
      { step: 2, kind: "deliver", tag: "ready", from: 2, to: 0, payload: { shard: 2 } },
      { step: 3, kind: "send", tag: "weights", from: 1, to: 0, payload: [4, 5] },
      { step: 4, kind: "deliver", tag: "weights", from: 1, to: 0, payload: [4, 5] },
      { step: 5, kind: "send", tag: "weights", from: 1, to: 2, payload: [4, 5] },
      { step: 6, kind: "deliver", tag: "weights", from: 1, to: 2, payload: [4, 5] },
      {
        step: 7,
        kind: "all-reduce",
        tag: "grad-bucket-0",
        op: "sum",
        participants: [0, 1, 2],
        values: [1, 2, 3],
        result: 6,
      },
    ]);
  });

  it("returns immutable trace snapshots", () => {
    const mesh = createDeterministicMesh({ ranks: 2 });
    mesh.barrier("sync");
    mesh.send({ from: 0, to: 1, tag: "payload", payload: { shard: { id: 1 } } });

    const snapshot = mesh.trace();
    const first = snapshot[0];
    if (!first?.participants) throw new Error("missing barrier participants");
    expect(first.participants).toEqual([0, 1]);
    first.participants.push(99);
    const sendPayload = snapshot[1]?.payload;
    if (!isShardPayload(sendPayload)) throw new Error("missing send payload");
    sendPayload.shard.id = 99;

    expect(mesh.trace()[0]?.participants).toEqual([0, 1]);
    expect((mesh.trace()[1]?.payload as { shard: { id: number } }).shard.id).toBe(1);
  });

  it("keeps public methods usable after destructuring", () => {
    const mesh = createDeterministicMesh({ ranks: 2 });
    const { broadcast } = mesh;

    broadcast({ from: 0, tag: "weights", payload: [1] });

    expect(mesh.trace().map((event) => event.kind)).toEqual(["send", "deliver"]);
  });

  it("rejects invalid ranks, reductions, and payload shapes", () => {
    expect(() => createDeterministicMesh({ ranks: 0 })).toThrow(SimulatorError);

    const mesh = createDeterministicMesh({ ranks: 2 });
    expect(() => mesh.send({ from: 0, to: 2, tag: "bad" })).toThrow(SimulatorError);
    expect(() =>
      mesh.allReduce({ tag: "bad", values: [1], op: "sum" }),
    ).toThrow(SimulatorError);
    expect(() =>
      mesh.allReduce({ tag: "bad", values: [1, Number.NaN], op: "sum" }),
    ).toThrow(SimulatorError);
  });
});

function isShardPayload(value: unknown): value is { shard: { id: number } } {
  return (
    typeof value === "object" &&
    value !== null &&
    "shard" in value &&
    typeof value.shard === "object" &&
    value.shard !== null &&
    "id" in value.shard &&
    typeof value.shard.id === "number"
  );
}
