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
  simulateVectorizedArraySum,
  simulateVectorizedClampedExp,
} from "../../browsergrad-simulators/src/index";
import { simulateCuda1DGrid } from "../../browsergrad-kernels/src/cuda_concepts";

interface CpuParallelismOracle {
  simulateVectorizedClampedExp: typeof simulateVectorizedClampedExp;
  simulateVectorizedArraySum: typeof simulateVectorizedArraySum;
  partitionStaticWork: typeof partitionStaticWork;
}

interface GpuPuzzleOracle {
  simulateCuda1DGrid: typeof simulateCuda1DGrid;
}

const CS149_A1_PROFILE = JSON.parse(
  readFileSync(
    new URL("../../../docs/internal/cs149-assignment1.profile.json", import.meta.url),
    "utf8",
  ),
);

const GPU_PUZZLES_PROFILE = JSON.parse(
  readFileSync(
    new URL("../../../docs/internal/gpu-puzzles.profile.json", import.meta.url),
    "utf8",
  ),
);

describe("profile-driven JavaScript assignment e2e", () => {
  it("rejects missing profile-declared JS oracles before running the rubric", async () => {
    const parsed = parseAssignmentProfile(GPU_PUZZLES_PROFILE);
    expect(parsed).toMatchObject({ ok: true });
    if (!parsed.ok) return;

    const environment = createAssignmentCapabilityEnvironment({
      browserCapabilities: ["kernel-visualizer", "webgpu", "wgsl-kernel"],
    });
    const contents = {
      files: {
        "/assignments/gpu-puzzles/rubric.js":
          "export default function rubric(ctx) { return ctx.id; }",
      },
    };
    let called = false;

    await expect(
      runAssignmentJavascriptProfile(parsed.profile, environment, contents, () => {
        called = true;
      }),
    ).rejects.toThrow(
      "missing declared JavaScript assignment oracle: _bg_cuda_concepts",
    );
    expect(called).toBe(false);
  });

  it("runs the GPU Puzzles profile with a CUDA-shaped JS oracle", async () => {
    const parsed = parseAssignmentProfile(GPU_PUZZLES_PROFILE);
    expect(parsed).toMatchObject({ ok: true });
    if (!parsed.ok) return;

    const environment = createAssignmentCapabilityEnvironment({
      browserCapabilities: ["kernel-visualizer", "webgpu", "wgsl-kernel"],
    });
    const contents = {
      files: {
        "/assignments/gpu-puzzles/rubric.js":
          "export default function rubric(ctx) { return ctx.id; }",
      },
    };
    const oracle: GpuPuzzleOracle = { simulateCuda1DGrid };
    const rubric: AssignmentJavascriptRubric = (ctx) => {
      const puzzles = ctx.oracle<GpuPuzzleOracle>("_bg_cuda_concepts");
      const map = puzzles.simulateCuda1DGrid({
        inputLength: 4,
        outputLength: 4,
        threadsPerBlock: 2,
        blocks: 2,
        initialInput: [0, 1, 2, 3],
        kernel(thread) {
          const i = thread.globalThreadId;
          thread.write(i, thread.read(i) + 10);
        },
      });
      const guard = puzzles.simulateCuda1DGrid({
        inputLength: 4,
        outputLength: 4,
        threadsPerBlock: 8,
        blocks: 1,
        initialInput: [0, 1, 2, 3],
        kernel(thread) {
          const i = thread.globalThreadId;
          if (i < thread.inputLength) {
            thread.write(i, thread.read(i) + 20);
          }
        },
      });

      if (
        map.output.join(",") !== "10,11,12,13" ||
        guard.output.join(",") !== "20,21,22,23" ||
        guard.violations.length !== 0
      ) {
        ctx.assertFail("gpu-puzzle-oracle", "GPU puzzle oracle mismatch", {
          expected: "map output and guarded kernel without OOB",
          actual: { map: map.output, guard: guard.output, guardViolations: guard.violations },
        });
        return;
      }

      ctx.assertPass("puzzle_map");
      ctx.assertPass("puzzle_guard");
      ctx.emitJson("gpu-puzzles-summary", {
        mapReads: map.stats.globalReads,
        guardThreads: guard.stats.launchedThreads,
        guardViolations: guard.violations.length,
      });
    };

    const result = await runAssignmentJavascriptProfile(
      parsed.profile,
      environment,
      contents,
      rubric,
      { oracles: { _bg_cuda_concepts: oracle } },
    );

    expect(result.report.runnerRoute.target).toBe("javascript");
    expect(result.report.readiness.status).toBe("runnable");
    expect(result.run.mount.writtenPaths).toEqual([
      "/assignments/gpu-puzzles/rubric.js",
    ]);
    expect(result.run.assertions).toEqual([
      { kind: "pass", name: "puzzle_map" },
      { kind: "pass", name: "puzzle_guard" },
    ]);
    expect(result.run.artifacts).toEqual([
      expect.objectContaining({
        kind: "json",
        name: "gpu-puzzles-summary",
        data: {
          mapReads: 4,
          guardThreads: 8,
          guardViolations: 0,
        },
      }),
    ]);
  });

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
    const oracle: CpuParallelismOracle = {
      simulateVectorizedClampedExp,
      simulateVectorizedArraySum,
      partitionStaticWork,
    };
    const rubric: AssignmentJavascriptRubric = (ctx) => {
      const cpuParallelism = ctx.oracle<CpuParallelismOracle>("_bg_cpu_parallelism");
      const fixtureText = ctx.readText("/assignments/cs149-assignment1/rubric.js");

      if (!ctx.allowedTests.includes("clamped_exp_simd")) {
        ctx.assertFail("allowed-tests", "profile did not expose clamped_exp_simd");
        return;
      }

      const clamped = cpuParallelism.simulateVectorizedClampedExp({
        values: [2, 3, -2, 4, 2],
        exponents: [0, 2, 3, 4, 5],
        vectorWidth: 4,
      });
      const summed = cpuParallelism.simulateVectorizedArraySum({
        values: [1, 2, 3, 4, 5, 6, 7, 8],
        vectorWidth: 4,
      });
      const partitions = cpuParallelism.partitionStaticWork({
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
      { oracles: { _bg_cpu_parallelism: oracle } },
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
