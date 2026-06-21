import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  assignmentRubricKind,
  createAssignmentCapabilityEnvironment,
  createAssignmentPreflightReport,
  createAssignmentRunPlan,
  evaluateAssignmentMountContents,
  parseAssignmentProfile,
  requiredAssignmentCapabilities,
  type AssignmentRunReadinessStatus,
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
    "performance-rubric",
    "pii-oracle",
    "pyodide",
    "rl-loss-oracle",
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
    });
  }
});
