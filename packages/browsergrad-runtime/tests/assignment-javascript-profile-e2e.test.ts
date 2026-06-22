import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  createAssignmentCapabilityEnvironment,
  parseAssignmentProfile,
  runAssignmentJavascriptProfile,
  runVerifiedAssignmentJavascriptProfile,
  type AssignmentJavascriptRubric,
} from "../src/index";
import { simulation } from "@unlocalhosted/browsergrad-primitives";
import {
  defineCuda1DProgram,
  emitCuda1DProgramWgsl,
  referenceExclusiveScan,
  referenceFindRepeats,
  referenceOrderedCircleRender,
  referenceSaxpy,
  runCuda1DProgramWebGpu,
  simulateCuda1DGrid,
  simulateCuda1DProgram,
} from "@unlocalhosted/browsergrad-kernels";
import {
  compileCudaLiteKernel,
  runCompiledKernelReference,
} from "@unlocalhosted/browsergrad-compiler";

interface CpuParallelismOracle {
  simulateVectorizedClampedExp: typeof simulation.simulateVectorizedClampedExp;
  simulateVectorizedArraySum: typeof simulation.simulateVectorizedArraySum;
  partitionStaticWork: typeof simulation.partitionStaticWork;
}

interface TaskGraphOracle {
  createTaskGraphSimulator: typeof simulation.createTaskGraphSimulator;
}

interface GpuPuzzleOracle {
  simulateCuda1DGrid: typeof simulateCuda1DGrid;
}

interface CudaConceptsReference {
  defineCuda1DProgram: typeof defineCuda1DProgram;
  emitCuda1DProgramWgsl: typeof emitCuda1DProgramWgsl;
  referenceExclusiveScan: typeof referenceExclusiveScan;
  referenceFindRepeats: typeof referenceFindRepeats;
  referenceOrderedCircleRender: typeof referenceOrderedCircleRender;
  referenceSaxpy: typeof referenceSaxpy;
  runCuda1DProgramWebGpu: typeof runCuda1DProgramWebGpu;
  simulateCuda1DGrid: typeof simulateCuda1DGrid;
  simulateCuda1DProgram: typeof simulateCuda1DProgram;
}

interface CudaLiteCompilerReference {
  compileCudaLiteKernel: typeof compileCudaLiteKernel;
  runCompiledKernelReference: typeof runCompiledKernelReference;
}

const CS149_A1_PROFILE = JSON.parse(
  readFileSync(
    new URL("../../../docs/internal/cs149-assignment1.profile.json", import.meta.url),
    "utf8",
  ),
);

const CS149_A2_PROFILE = JSON.parse(
  readFileSync(
    new URL("../../../docs/internal/cs149-assignment2.profile.json", import.meta.url),
    "utf8",
  ),
);

const CS149_A3_PROFILE = JSON.parse(
  readFileSync(
    new URL("../../../docs/internal/cs149-assignment3.profile.json", import.meta.url),
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

  it("blocks verified JavaScript profile runs before rubric execution when dataset hashes drift", async () => {
    const parsed = parseAssignmentProfile({
      ...GPU_PUZZLES_PROFILE,
      datasets: [
        {
          name: "cases",
          url: "https://example.invalid/gpu-puzzles/cases.txt",
          hash:
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        },
      ],
    });
    expect(parsed).toMatchObject({ ok: true });
    if (!parsed.ok) return;

    const environment = createAssignmentCapabilityEnvironment({
      browserCapabilities: ["kernel-visualizer", "webgpu", "wgsl-kernel"],
    });
    const files = {
      "/assignments/gpu-puzzles/rubric.js":
        "export default function rubric(ctx) { return ctx.id; }",
    };
    const oracle: GpuPuzzleOracle = { simulateCuda1DGrid };
    let called = false;
    const rubric: AssignmentJavascriptRubric = () => {
      called = true;
    };

    await expect(
      runVerifiedAssignmentJavascriptProfile(
        parsed.profile,
        environment,
        { files, datasets: { cases: "abcd" } },
        rubric,
        { oracles: { _bg_cuda_concepts: oracle } },
      ),
    ).rejects.toThrow("dataset hash mismatch: cases");
    expect(called).toBe(false);

    await expect(
      runVerifiedAssignmentJavascriptProfile(
        parsed.profile,
        environment,
        { files, datasets: { cases: "abc" } },
        rubric,
        { oracles: { _bg_cuda_concepts: oracle } },
      ),
    ).resolves.toMatchObject({
      run: { assertions: [], artifacts: [] },
    });
    expect(called).toBe(true);
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
      simulateVectorizedClampedExp: simulation.simulateVectorizedClampedExp,
      simulateVectorizedArraySum: simulation.simulateVectorizedArraySum,
      partitionStaticWork: simulation.partitionStaticWork,
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

  it("runs the CS149 A2 profile with a generic task-graph simulator", async () => {
    const parsed = parseAssignmentProfile(CS149_A2_PROFILE);
    expect(parsed).toMatchObject({ ok: true });
    if (!parsed.ok) return;

    const environment = createAssignmentCapabilityEnvironment({
      browserCapabilities: ["performance-rubric"],
      simulatedCapabilities: [
        "pthreads-simulator",
        "task-graph-simulator",
        "worker-mesh",
      ],
    });
    const contents = {
      files: {
        "/assignments/cs149-assignment2/rubric.js":
          "export default async function rubric(ctx) { return ctx.id; }",
      },
    };
    const oracle: TaskGraphOracle = {
      createTaskGraphSimulator: simulation.createTaskGraphSimulator,
    };
    const rubric: AssignmentJavascriptRubric = (ctx) => {
      const taskGraph = ctx.oracle<TaskGraphOracle>("_bg_task_graph");
      const simulator = taskGraph.createTaskGraphSimulator({ workers: 2 });
      simulator.addTask({ id: "root", duration: 2 });
      simulator.addTask({ id: "left", duration: 3, dependsOn: ["root"] });
      simulator.addTask({ id: "right", duration: 4, dependsOn: ["root"] });
      const run = simulator.run();

      if (
        !ctx.allowedTests.includes("task_graph_dependencies") ||
        run.makespan !== 6 ||
        run.completedTaskIds.join(",") !== "root,left,right"
      ) {
        ctx.assertFail("cs149-a2-task-graph", "CS149 A2 task graph mismatch", {
          expected: {
            makespan: 6,
            completedTaskIds: ["root", "left", "right"],
          },
          actual: run,
        });
        return;
      }

      ctx.assertPass("task_graph_dependencies");
      ctx.emitJson("cs149-a2-summary", {
        readiness: "simulated",
        makespan: run.makespan,
        starts: run.events.filter((event) => event.kind === "task-start").length,
      });
    };

    const result = await runAssignmentJavascriptProfile(
      parsed.profile,
      environment,
      contents,
      rubric,
      { oracles: { _bg_task_graph: oracle } },
    );

    expect(result.report.runnerRoute.target).toBe("javascript");
    expect(result.report.readiness.status).toBe("simulated");
    expect(result.run.assertions).toEqual([
      { kind: "pass", name: "task_graph_dependencies" },
    ]);
    expect(result.run.artifacts).toEqual([
      expect.objectContaining({
        kind: "json",
        name: "cs149-a2-summary",
        data: {
          readiness: "simulated",
          makespan: 6,
          starts: 3,
        },
      }),
    ]);
  });

  it("runs the CS149 A3 profile with generic CUDA concept references", async () => {
    const parsed = parseAssignmentProfile(CS149_A3_PROFILE);
    expect(parsed).toMatchObject({ ok: true });
    if (!parsed.ok) return;

    const environment = createAssignmentCapabilityEnvironment({
      browserCapabilities: [
        "cuda-compatible-subset",
        "performance-rubric",
        "webgpu",
        "wgsl-kernel",
      ],
    });
    const contents = {
      files: {
        "/assignments/cs149-assignment3/rubric.js":
          "export default async function rubric(ctx) { return ctx.id; }",
      },
    };
    const reference: CudaConceptsReference = {
      defineCuda1DProgram,
      emitCuda1DProgramWgsl,
      referenceExclusiveScan,
      referenceFindRepeats,
      referenceOrderedCircleRender,
      referenceSaxpy,
      runCuda1DProgramWebGpu,
      simulateCuda1DGrid,
      simulateCuda1DProgram,
    };
    const rubric: AssignmentJavascriptRubric = (ctx) => {
      const cuda = ctx.oracle<CudaConceptsReference>("_bg_cuda_concepts");
      const saxpy = cuda.referenceSaxpy({
        a: 2,
        x: [1, 2, 3, 4],
        y: [10, 20, 30, 40],
      });
      const scan = cuda.referenceExclusiveScan([3, 1, 4, 1, 5]);
      const repeats = cuda.referenceFindRepeats([7, 7, 1, 4, 4, 4]);
      const saxpyProgram = cuda.defineCuda1DProgram({
        name: "saxpy",
        inputLength: 4,
        outputLength: 4,
        parameters: { a: 2 },
        launch: { blocks: 1, threadsPerBlock: 8 },
        body: [
          {
            op: "if",
            condition: {
              op: "lt",
              left: { op: "threadId" },
              right: { op: "outputLength" },
            },
            body: [
              {
                op: "write",
                index: { op: "threadId" },
                value: {
                  op: "add",
                  left: {
                    op: "mul",
                    left: { op: "param", name: "a" },
                    right: { op: "read", index: { op: "threadId" } },
                  },
                  right: { op: "outputRead", index: { op: "threadId" } },
                },
              },
            ],
          },
        ],
      });
      const loweredSaxpy = cuda.simulateCuda1DProgram(saxpyProgram, {
        initialInput: [1, 2, 3, 4],
        initialOutput: [10, 20, 30, 40],
      });
      const wgsl = cuda.emitCuda1DProgramWgsl(saxpyProgram);
      const rendered = cuda.referenceOrderedCircleRender({
        width: 1,
        height: 1,
        background: [0, 0, 0],
        circles: [
          { center: [0.5, 0.5], radius: 1, color: [1, 0, 0], alpha: 0.5 },
          { center: [0.5, 0.5], radius: 1, color: [0, 1, 0], alpha: 0.5 },
          { center: [0.5, 0.5], radius: 1, color: [0, 0, 1], alpha: 0.5 },
        ],
      });
      const guard = cuda.simulateCuda1DGrid({
        inputLength: 4,
        outputLength: 4,
        threadsPerBlock: 8,
        blocks: 1,
        initialInput: [0, 1, 2, 3],
        kernel(thread) {
          const i = thread.globalThreadId;
          if (i < thread.inputLength) {
            thread.write(i, thread.read(i) + 1);
          }
        },
      });

      if (
        !ctx.allowedTests.includes("saxpy_correctness") ||
        !ctx.allowedTests.includes("wgsl_lowering_smoke") ||
        saxpy.join(",") !== "12,24,36,48" ||
        loweredSaxpy.output.join(",") !== "12,24,36,48" ||
        typeof cuda.runCuda1DProgramWebGpu !== "function" ||
        scan.join(",") !== "0,3,4,8,9" ||
        repeats.join(",") !== "0,3,4" ||
        rendered.pixels.join(",") !== "0.125,0.25,0.5" ||
        guard.violations.length !== 0
      ) {
        ctx.assertFail("cs149-a3-cuda-concepts", "CS149 A3 CUDA concept mismatch", {
          expected: {
            saxpy: [12, 24, 36, 48],
            loweredSaxpy: [12, 24, 36, 48],
            webgpuRunner: "function",
            scan: [0, 3, 4, 8, 9],
            repeats: [0, 3, 4],
            renderedPixel: [0.125, 0.25, 0.5],
            guardViolations: 0,
          },
          actual: {
            saxpy,
            loweredSaxpy: loweredSaxpy.output,
            webgpuRunner: typeof cuda.runCuda1DProgramWebGpu,
            scan,
            repeats,
            renderedPixel: rendered.pixels,
            guardViolations: guard.violations,
          },
        });
        return;
      }

      ctx.assertPass("saxpy_correctness");
      ctx.assertPass("wgsl_lowering_smoke");
      ctx.assertPass("exclusive_scan_correctness");
      ctx.assertPass("renderer_ordering_correctness");
      ctx.assertPass("kernel_memory_bounds");
      ctx.emitJson("cs149-a3-summary", {
        readiness: "browser-kernel-concepts",
        guardThreads: guard.stats.launchedThreads,
        globalWrites: guard.stats.globalWrites,
        renderedPixel: rendered.pixels,
        repeatCount: repeats.length,
        wgslHasParams: wgsl.includes("var<uniform> params"),
      });
    };

    const result = await runAssignmentJavascriptProfile(
      parsed.profile,
      environment,
      contents,
      rubric,
      { oracles: { _bg_cuda_concepts: reference } },
    );

    expect(result.report.runnerRoute.target).toBe("javascript");
    expect(result.report.readiness.status).toBe("runnable");
    expect(result.run.assertions).toEqual([
      { kind: "pass", name: "saxpy_correctness" },
      { kind: "pass", name: "wgsl_lowering_smoke" },
      { kind: "pass", name: "exclusive_scan_correctness" },
      { kind: "pass", name: "renderer_ordering_correctness" },
      { kind: "pass", name: "kernel_memory_bounds" },
    ]);
    expect(result.run.artifacts).toEqual([
      expect.objectContaining({
        kind: "json",
        name: "cs149-a3-summary",
        data: {
          readiness: "browser-kernel-concepts",
          guardThreads: 8,
          globalWrites: 4,
          renderedPixel: [0.125, 0.25, 0.5],
          repeatCount: 3,
          wgslHasParams: true,
        },
      }),
    ]);
  });

  it("runs the CS149 A3 profile through the CUDA-lite compiler capability", async () => {
    const parsed = parseAssignmentProfile(CS149_A3_PROFILE);
    expect(parsed).toMatchObject({ ok: true });
    if (!parsed.ok) return;

    const environment = createAssignmentCapabilityEnvironment({
      browserCapabilities: [
        "cuda-compatible-subset",
        "cuda-lite-compiler",
        "performance-rubric",
        "webgpu",
      ],
    });
    const contents = {
      files: {
        "/assignments/cs149-assignment3/rubric.js":
          "export default async function rubric(ctx) { return ctx.id; }",
      },
    };
    const compiler: CudaLiteCompilerReference = {
      compileCudaLiteKernel,
      runCompiledKernelReference,
    };
    const rubric: AssignmentJavascriptRubric = (ctx) => {
      const cuda = ctx.oracle<CudaLiteCompilerReference>("_bg_cuda_concepts");
      const compiled = cuda.compileCudaLiteKernel(
        `
__global__ void saxpy(const float* x, float* y, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}
`,
        { workgroupSize: [4, 1, 1] },
      );
      const result = cuda.runCompiledKernelReference(
        compiled,
        {
          buffers: {
            x: new Float32Array([1, 2, 3, 4]),
            y: new Float32Array([10, 20, 30, 40]),
          },
          scalars: { a: 2, n: 4 },
        },
        { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      );
      const y = Array.from(result.buffers.y as Float32Array);

      if (
        !ctx.allowedTests.includes("saxpy_correctness") ||
        !ctx.allowedTests.includes("performance_rubric_smoke") ||
        y.join(",") !== "12,24,36,48" ||
        !compiled.wgsl.includes("@compute")
      ) {
        ctx.assertFail(
          "cs149-a3-cuda-lite-compiler",
          "CUDA-lite compiler oracle mismatch",
          {
            expected: { y: [12, 24, 36, 48], wgsl: "@compute" },
            actual: { y, wgslPrefix: compiled.wgsl.slice(0, 80) },
          },
        );
        return;
      }

      ctx.assertPass("saxpy_correctness");
      ctx.assertPass("wgsl_lowering_smoke");
      ctx.assertPass("performance_rubric_smoke");
      ctx.emitJson("cs149-a3-cuda-lite-summary", {
        selectedPath: "cuda-lite-compiler",
        output: y,
        bindingCount: compiled.wgslProgram.bindings.length,
      });
    };

    const result = await runAssignmentJavascriptProfile(
      parsed.profile,
      environment,
      contents,
      rubric,
      { oracles: { _bg_cuda_concepts: compiler } },
    );

    expect(result.report.runnerRoute.target).toBe("javascript");
    expect(result.report.readiness.status).toBe("runnable");
    expect(result.report.readiness.selectedCapabilities).toEqual([
      "cuda-compatible-subset",
      "cuda-lite-compiler",
      "performance-rubric",
      "webgpu",
    ]);
    expect(result.run.mount).toEqual({
      writtenPaths: ["/assignments/cs149-assignment3/rubric.js"],
      skippedOptionalPaths: [
        "/assignments/cs149-assignment3/starter",
        "/assignments/cs149-assignment3/reference.js",
      ],
    });
    expect(result.run.assertions).toEqual([
      { kind: "pass", name: "saxpy_correctness" },
      { kind: "pass", name: "wgsl_lowering_smoke" },
      { kind: "pass", name: "performance_rubric_smoke" },
    ]);
    expect(result.run.artifacts).toEqual([
      expect.objectContaining({
        kind: "json",
        name: "cs149-a3-cuda-lite-summary",
        data: {
          selectedPath: "cuda-lite-compiler",
          output: [12, 24, 36, 48],
          bindingCount: 3,
        },
      }),
    ]);
  });
});
