import { describe, expect, it } from "vitest";

import {
  createByteBpeReference,
  createDataCleaningReference,
  createHostedTrainingApiFixture,
  createSnapshotComparator,
  data,
  evaluation,
  rl,
  scaling,
  simulation,
  text,
} from "../src/index";
import { createByteBpeReference as createByteBpeReferenceFromText } from "../src/text";

describe("@unlocalhosted/browsergrad-primitives public surface", () => {
  it("exposes generic primitive namespaces instead of lab-shaped package names", () => {
    expect(text.trainByteBpe).toBe(createByteBpeReference().trainByteBpe);
    expect(typeof data.extractVisibleTextFromHtml).toBe("function");
    expect(typeof evaluation.createSnapshotComparator).toBe("function");
    expect(typeof scaling.createHostedTrainingApiFixture).toBe("function");
    expect(typeof simulation.createDeterministicMesh).toBe("function");
    expect(typeof rl.computePerInstanceDpoLoss).toBe("function");
  });

  it("uses text references for byte-BPE behavior", () => {
    const reference = createByteBpeReferenceFromText();
    const model = reference.trainByteBpe("low low lower", { vocabSize: 258 });

    expect(reference.decodeByteBpe(reference.encodeByteBpe("low", model), model)).toBe(
      "low",
    );
  });

  it("uses evaluation comparators without rubric/oracle vocabulary", () => {
    const comparator = createSnapshotComparator(
      { loss: 1.25, logits: [0.1, 0.2] },
      { absoluteTolerance: 0.01 },
    );

    expect(comparator.compare({ loss: 1.251, logits: [0.1, 0.205] }).ok).toBe(
      true,
    );
    expect(comparator.compare({ loss: 1.4, logits: [0.1, 0.2] }).ok).toBe(false);
  });

  it("wraps hosted training fixtures with userId vocabulary", () => {
    const api = createHostedTrainingApiFixture({ totalBudgetSeconds: 30 });
    api.registerUser({ userId: "alice", apiKey: "test-key" });
    const submitted = api.submitExperiment("test-key", {
      model: "tiny-transformer",
      max_runtime_seconds: 10,
    });

    expect(submitted.budgetSummary.remainingSeconds).toBe(20);
    expect(api.listExperiments("test-key")[0]).toMatchObject({
      userId: "alice",
      trainingConfig: { model: "tiny-transformer", max_runtime_seconds: 10 },
    });
  });

  it("creates data-cleaning references without rubric adapter vocabulary", () => {
    const reference = createDataCleaningReference();

    expect(reference.extractVisibleTextFromHtml("<p>Hello&nbsp;data</p>")).toBe(
      "Hello data",
    );
    expect(reference.maskPii("Email jane@example.com").text).toBe("Email <EMAIL>");
    expect(
      reference.exactLineDeduplicate(["alpha", "beta", "alpha"]).keptLines,
    ).toEqual(["alpha", "beta"]);
    expect(
      reference.minhashDeduplicateDocuments([
        { id: "a", text: "permission is hereby granted free of charge" },
        { id: "b", text: "permission is hereby granted free of charge" },
      ]).duplicates.length,
    ).toBe(1);
    expect(reference.evaluateGopherQuality("high quality words ".repeat(80)).passed)
      .toBe(true);
  });
});
