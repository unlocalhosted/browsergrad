import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  assignmentRubricKind,
  createAssignmentCapabilityEnvironment,
  createAssignmentBenchmarkPreflightMatrix,
  createAssignmentDatasetCachePlan,
  createAssignmentPreflightReport,
  createAssignmentRunPlan,
  evaluateAssignmentMountContents,
  parseAssignmentProfile,
  requiredAssignmentCapabilities,
  type AssignmentRunReadinessStatus,
  type AssignmentRunnerTarget,
} from "../src/index";

const PROFILE_FILES = [
  "cs336-assignment1.profile.json",
  "cs336-assignment2-systems.profile.json",
  "cs336-assignment3-scaling.profile.json",
  "cs336-assignment4-data.profile.json",
  "cs336-assignment5-alignment.profile.json",
  "gpu-puzzles.profile.json",
  "cs149-assignment1.profile.json",
  "cs149-assignment2.profile.json",
  "cs149-assignment3.profile.json",
  "cs149gpt.profile.json",
];

const EXPECTED_RUBRIC_KIND: Record<string, "python" | "javascript"> = {
  "cs336-assignment1.profile.json": "python",
  "cs336-assignment2-systems.profile.json": "python",
  "cs336-assignment3-scaling.profile.json": "python",
  "cs336-assignment4-data.profile.json": "python",
  "cs336-assignment5-alignment.profile.json": "python",
  "gpu-puzzles.profile.json": "javascript",
  "cs149-assignment1.profile.json": "javascript",
  "cs149-assignment2.profile.json": "javascript",
  "cs149-assignment3.profile.json": "javascript",
  "cs149gpt.profile.json": "python",
};

const BROWSER_TEACHING_ENVIRONMENT = createAssignmentCapabilityEnvironment({
  browserCapabilities: [
    "attention-oracle",
    "classifier-oracle",
    "dataset-fixture",
    "dedupe-oracle",
    "http-client",
    "js-oracles",
    "kernel-visualizer",
    "large-file-streaming",
    "near-dedupe-oracle",
    "performance-rubric",
    "pii-oracle",
    "quality-rule-oracle",
    "pyodide",
    "response-parser-oracle",
    "rl-loss-oracle",
    "scaling-law-oracle",
    "server-fixture",
    "snapshot-oracle",
    "structured-assertions",
    "tokenizer-oracle",
    "torch-compat",
    "webgpu",
    "wgsl-kernel",
  ],
  simulatedCapabilities: [
    "browser-cpp-simulator",
    "browser-math-only",
    "cuda-compatible-subset",
    "distributed-simulator",
    "hosted-api-mock",
    "ispc-simulator",
    "pthreads-simulator",
    "scheduler-simulator",
    "simd-simulator",
    "task-graph-simulator",
    "worker-mesh",
  ],
});

const EXPECTED_BROWSER_TEACHING_READINESS: Record<
  string,
  AssignmentRunReadinessStatus
> = {
  "cs336-assignment1.profile.json": "runnable",
  "cs336-assignment2-systems.profile.json": "simulated",
  "cs336-assignment3-scaling.profile.json": "simulated",
  "cs336-assignment4-data.profile.json": "runnable",
  "cs336-assignment5-alignment.profile.json": "simulated",
  "gpu-puzzles.profile.json": "runnable",
  "cs149-assignment1.profile.json": "simulated",
  "cs149-assignment2.profile.json": "simulated",
  "cs149-assignment3.profile.json": "simulated",
  "cs149gpt.profile.json": "simulated",
};

const EXPECTED_BROWSER_TEACHING_RUNNER: Record<string, AssignmentRunnerTarget> = {
  "cs336-assignment1.profile.json": "pyodide",
  "cs336-assignment2-systems.profile.json": "pyodide",
  "cs336-assignment3-scaling.profile.json": "pyodide",
  "cs336-assignment4-data.profile.json": "pyodide",
  "cs336-assignment5-alignment.profile.json": "pyodide",
  "gpu-puzzles.profile.json": "javascript",
  "cs149-assignment1.profile.json": "javascript",
  "cs149-assignment2.profile.json": "javascript",
  "cs149-assignment3.profile.json": "javascript",
  "cs149gpt.profile.json": "pyodide",
};

const EXTERNAL_BENCHMARK_ENVIRONMENTS = {
  "cs336-assignment5-alignment.profile.json": {
    environment: createAssignmentCapabilityEnvironment({
      browserCapabilities: [
        "pyodide",
        "torch-compat",
        "snapshot-oracle",
        "tokenizer-oracle",
        "rl-loss-oracle",
        "response-parser-oracle",
      ],
      externalCapabilities: ["vllm-external", "flash-attn-external"],
    }),
    externalCapabilities: ["flash-attn-external", "vllm-external"],
  },
  "cs149gpt.profile.json": {
    environment: createAssignmentCapabilityEnvironment({
      browserCapabilities: ["attention-oracle", "performance-rubric"],
      simulatedCapabilities: ["simd-simulator"],
      externalCapabilities: ["native-cpp-external"],
    }),
    externalCapabilities: ["native-cpp-external"],
  },
} as const;

describe("benchmark assignment profiles", () => {
  for (const file of PROFILE_FILES) {
    it(`parses ${file} and declares capability requirements`, () => {
      const profileJson = JSON.parse(
        readFileSync(new URL(`../../../docs/internal/${file}`, import.meta.url), "utf8"),
      );

      const result = parseAssignmentProfile(profileJson);
      expect(result).toMatchObject({ ok: true });
      if (!result.ok) return;

      expect(result.profile.metadata?.source_url).toMatch(/^https:\/\/github.com\//);
      expect(requiredAssignmentCapabilities(result.profile).length).toBeGreaterThan(0);
      expect(
        assignmentRubricKind(
          createAssignmentRunPlan(result.profile, {
            capabilities: requiredAssignmentCapabilities(result.profile),
          }),
        ),
      ).toBe(EXPECTED_RUBRIC_KIND[file]);
    });

    it(`creates browser-teaching readiness for ${file}`, () => {
      const profileJson = JSON.parse(
        readFileSync(new URL(`../../../docs/internal/${file}`, import.meta.url), "utf8"),
      );

      const result = parseAssignmentProfile(profileJson);
      expect(result).toMatchObject({ ok: true });
      if (!result.ok) return;

      const report = createAssignmentPreflightReport(
        result.profile,
        BROWSER_TEACHING_ENVIRONMENT,
      );

      expect(report.readiness.status).toBe(EXPECTED_BROWSER_TEACHING_READINESS[file]);
      expect(report.runnerRoute.target).toBe(EXPECTED_BROWSER_TEACHING_RUNNER[file]);
      expect(report.requiredCapabilities).toEqual(
        requiredAssignmentCapabilities(result.profile),
      );
      expect(report.mountPlan.root).toBe(result.profile.files.root);
      expect(report.rubricKind).toBe(EXPECTED_RUBRIC_KIND[file]);
    });

    it(`dry-runs missing mount contents for ${file}`, () => {
      const profileJson = JSON.parse(
        readFileSync(new URL(`../../../docs/internal/${file}`, import.meta.url), "utf8"),
      );

      const result = parseAssignmentProfile(profileJson);
      expect(result).toMatchObject({ ok: true });
      if (!result.ok) return;

      const report = createAssignmentPreflightReport(
        result.profile,
        BROWSER_TEACHING_ENVIRONMENT,
      );
      const contentEvaluation = evaluateAssignmentMountContents(report.mountPlan, {
        files: {},
      });

      expect(contentEvaluation.ok).toBe(false);
      expect(contentEvaluation.missingRequiredFiles).toEqual([
        report.plan.files.rubricPath,
      ]);
      expect(contentEvaluation.missingDatasets).toEqual(
        result.profile.datasets.map((dataset) => dataset.name),
      );

      const cachePlan = createAssignmentDatasetCachePlan(report.mountPlan);
      expect(cachePlan.datasets.map((dataset) => dataset.name)).toEqual(
        result.profile.datasets.map((dataset) => dataset.name),
      );
    });
  }

  it("creates one platform handoff matrix row per benchmark profile", () => {
    const profiles = PROFILE_FILES.map((file) => {
      const profileJson = JSON.parse(
        readFileSync(new URL(`../../../docs/internal/${file}`, import.meta.url), "utf8"),
      );
      const result = parseAssignmentProfile(profileJson);
      expect(result).toMatchObject({ ok: true });
      if (!result.ok) throw new Error(`${file} did not parse`);
      return result.profile;
    });

    const matrix = createAssignmentBenchmarkPreflightMatrix(
      profiles,
      BROWSER_TEACHING_ENVIRONMENT,
    );

    expect(matrix.ok).toBe(false);
    expect(matrix.rows.map((row) => row.id)).toEqual(
      profiles.map((profile) => profile.id),
    );
    expect(matrix.rows.map((row) => row.readinessStatus)).toEqual(
      PROFILE_FILES.map((file) => EXPECTED_BROWSER_TEACHING_READINESS[file]),
    );
    expect(matrix.rows.map((row) => row.runnerTarget)).toEqual(
      PROFILE_FILES.map((file) => EXPECTED_BROWSER_TEACHING_RUNNER[file]),
    );
    expect(matrix.rows.every((row) => row.contentOk === false)).toBe(true);
    expect(matrix.rows.every((row) => row.missingRequiredFiles.length === 1)).toBe(true);
    expect(matrix.rows.every((row) => row.externalRunnerRequired === false)).toBe(true);
    expect(matrix.rows.every((row) => row.gates.length > 0)).toBe(true);
    expect(matrix.rows.flatMap((row) => row.gates).every((gate) => gate.ok)).toBe(true);
  });

  it("creates external runner requests for native-heavy benchmark profiles", () => {
    for (const [file, expected] of Object.entries(EXTERNAL_BENCHMARK_ENVIRONMENTS)) {
      const profileJson = JSON.parse(
        readFileSync(new URL(`../../../docs/internal/${file}`, import.meta.url), "utf8"),
      );

      const result = parseAssignmentProfile(profileJson);
      expect(result).toMatchObject({ ok: true });
      if (!result.ok) continue;

      const report = createAssignmentPreflightReport(
        result.profile,
        expected.environment,
      );

      expect(report.runnerRoute.target).toBe("external");
      expect(report.externalRunnerRequest?.externalCapabilities).toEqual(
        expected.externalCapabilities,
      );
      expect(report.externalRunnerRequest?.files.root).toBe(result.profile.files.root);
      expect(report.externalRunnerRequest?.mountPlan).toEqual(report.mountPlan);
      expect(report.externalRunnerRequest?.datasetCachePlan).toEqual(
        report.datasetCachePlan,
      );
    }
  });
});
