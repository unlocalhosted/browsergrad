import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  createAssignmentCapabilityEnvironment,
  parseAssignmentProfile,
  runAssignmentJavascriptProfile,
  type AssignmentJavascriptRubric,
} from "../src/index";
import {
  partitionStaticWork,
  simulateCs149ArraySumVector,
  simulateCs149ClampedExpVector,
} from "../../browsergrad-simulators/src/index";

interface Cs149CpuOracle {
  simulateCs149ClampedExpVector: typeof simulateCs149ClampedExpVector;
  simulateCs149ArraySumVector: typeof simulateCs149ArraySumVector;
  partitionStaticWork: typeof partitionStaticWork;
}

const CS149_A1_PROFILE = JSON.parse(
  readFileSync(
    new URL("../../../docs/internal/cs149-assignment1.profile.json", import.meta.url),
    "utf8",
  ),
);

describe("profile-driven JavaScript assignment e2e", () => {
  it("runs the CS149 A1 profile through preflight, mounts, declared oracles, and JS assertions", async () => {
    const parsed = parseAssignmentProfile(CS149_A1_PROFILE);
    expect(parsed).toMatchObject({ ok: true });
    if (!parsed.ok) return;

    const environment = createAssignmentCapabilityEnvironment({
      browserCapabilities: ["performance-rubric"],
      simulatedCapabilities: [
        "ispc-simulator",
        "pthreads-simulator",
        "simd-simulator",
      ],
    });
    const contents = {
      files: {
        "/assignments/cs149-assignment1/rubric.js":
          "export default async function rubric(ctx) { return ctx.id; }",
      },
    };
    const oracle: Cs149CpuOracle = {
      simulateCs149ClampedExpVector,
      simulateCs149ArraySumVector,
      partitionStaticWork,
    };
    const rubric: AssignmentJavascriptRubric = (ctx) => {
      const cs149 = ctx.oracle<Cs149CpuOracle>("_bg_cs149_cpu_oracles");
      const fixtureText = ctx.readText("/assignments/cs149-assignment1/rubric.js");

      if (!ctx.allowedTests.includes("clamped_exp_simd")) {
        ctx.assertFail("allowed-tests", "profile did not expose clamped_exp_simd");
        return;
      }

      const clamped = cs149.simulateCs149ClampedExpVector({
        values: [2, 3, -2, 4, 2],
        exponents: [0, 2, 3, 4, 5],
        vectorWidth: 4,
      });
      const summed = cs149.simulateCs149ArraySumVector({
        values: [1, 2, 3, 4, 5, 6, 7, 8],
        vectorWidth: 4,
      });
      const partitions = cs149.partitionStaticWork({
        items: 10,
        workers: 3,
      });

      if (clamped.output[3] !== 9.999999 || summed.sum !== 36) {
        ctx.assertFail("cs149-a1-oracle", "CS149 A1 simulator oracle mismatch", {
          expected: "clamped output clamps lane 3 and array sum is 36",
          actual: { clamped: clamped.output, sum: summed.sum },
        });
        return;
      }

      ctx.assertPass("clamped_exp_simd");
      ctx.assertPass("array_sum_simd");
      ctx.assertPass("mandelbrot_thread_decomposition");
      ctx.log("mounted-rubric", fixtureText, "info");
      ctx.emitJson("cs149-a1-summary", {
        readiness: "simulated",
        clampedUtilization: clamped.stats.utilization,
        horizontalReductionRounds: summed.horizontalReductionRounds,
        partitions,
      });
    };

    const result = await runAssignmentJavascriptProfile(
      parsed.profile,
      environment,
      contents,
      rubric,
      { oracles: { _bg_cs149_cpu_oracles: oracle } },
    );

    expect(result.report.runnerRoute.target).toBe("javascript");
    expect(result.report.readiness.status).toBe("simulated");
    expect(result.run.mount.writtenPaths).toEqual([
      "/assignments/cs149-assignment1/rubric.js",
    ]);
    expect(result.run.assertions).toEqual([
      { kind: "pass", name: "clamped_exp_simd" },
      { kind: "pass", name: "array_sum_simd" },
      { kind: "pass", name: "mandelbrot_thread_decomposition" },
    ]);
    expect(result.run.artifacts).toEqual([
      expect.objectContaining({
        kind: "log",
        name: "mounted-rubric",
        level: "info",
      }),
      expect.objectContaining({
        kind: "json",
        name: "cs149-a1-summary",
        data: expect.objectContaining({
          readiness: "simulated",
          horizontalReductionRounds: 2,
        }),
      }),
    ]);
  });
});
