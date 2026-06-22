import { describe, expect, it } from "vitest";
import { VALID_PROFILE } from "./assignment-fixtures";
import {
  createAssignmentCapabilityEnvironment,
  createAssignmentCapabilityCatalog,
  createAssignmentBenchmarkPreflightMatrix,
  createVerifiedAssignmentBenchmarkPreflightMatrix,
  createAssignmentPlatformHandoff,
  createAssignmentPlatformIssueDraft,
  createVerifiedAssignmentPlatformHandoff,
  createAssignmentPreflightReport,
  parseAssignmentProfile,
} from "../src/index";

describe("assignment platform handoff", () => {
  it("creates a platform-ready benchmark preflight matrix", () => {
    const pythonResult = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "python_runtime",
          kind: "capability",
          options: { requires: ["pyodide"] },
        },
      ],
    });
    const jsResult = parseAssignmentProfile({
      ...VALID_PROFILE,
      id: "gpu-puzzles-smoke",
      files: {
        root: "/assignments/gpu-puzzles-smoke",
        rubric_path: "rubric.js",
        starter_path: "student.js",
        fixtures_path: "fixtures",
      },
      gates: [
        {
          name: "kernel_path",
          kind: "capability",
          options: {
            any_of: [["webgpu", "wgsl-kernel"], ["cuda-compatible-subset"]],
          },
        },
      ],
      datasets: [
        {
          name: "cases",
          url: "/fixtures/cases.bin",
          hash: "sha256:" + "a".repeat(64),
        },
      ],
    });
    expect(pythonResult.ok).toBe(true);
    expect(jsResult.ok).toBe(true);
    if (!pythonResult.ok || !jsResult.ok) return;

    const matrix = createAssignmentBenchmarkPreflightMatrix(
      [pythonResult.profile, jsResult.profile],
      createAssignmentCapabilityEnvironment({
        browserCapabilities: ["pyodide", "webgpu", "wgsl-kernel"],
      }),
      {
        files: {
          "/assignments/cs336-assignment1/rubric.py": "print('ok')",
        },
      },
    );

    expect(matrix.ok).toBe(false);
    expect(matrix.rows).toEqual([
      expect.objectContaining({
        id: "cs336-assignment1",
        title: "Stanford CS336 Assignment 1: Basics",
        readinessStatus: "runnable",
        runnerTarget: "pyodide",
        rubricKind: "python",
        contentOk: false,
        missingRequiredFiles: [],
        missingDatasets: ["tiny"],
        cacheStrategies: ["invalid-hash"],
        externalRunnerRequired: false,
        gates: [
          {
            name: "python_runtime",
            status: "runnable",
            ok: true,
            requires: ["pyodide"],
            anyOf: [],
            selectedAnyOf: [],
            selectedCapabilities: ["pyodide"],
            missingRequired: [],
            missingAnyOf: [],
          },
        ],
      }),
      expect.objectContaining({
        id: "gpu-puzzles-smoke",
        readinessStatus: "runnable",
        runnerTarget: "javascript",
        rubricKind: "javascript",
        contentOk: false,
        missingRequiredFiles: [
          "/assignments/gpu-puzzles-smoke/rubric.js",
        ],
        missingDatasets: ["cases"],
        cacheStrategies: ["content-addressed"],
        externalRunnerRequired: false,
        gates: [
          {
            name: "kernel_path",
            status: "runnable",
            ok: true,
            requires: [],
            anyOf: [["webgpu", "wgsl-kernel"], ["cuda-compatible-subset"]],
            selectedAnyOf: ["webgpu", "wgsl-kernel"],
            selectedCapabilities: ["webgpu", "wgsl-kernel"],
            missingRequired: [],
            missingAnyOf: [],
          },
        ],
      }),
    ]);
  });

  it("creates a platform handoff with the next action for launch UI", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "python_runtime",
          kind: "capability",
          options: { requires: ["pyodide"] },
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const environment = createAssignmentCapabilityEnvironment({
      browserCapabilities: ["pyodide"],
    });
    const report = createAssignmentPreflightReport(result.profile, environment);

    expect(
      createAssignmentPlatformHandoff(result.profile, report, { files: {} }),
    ).toEqual({
      id: "cs336-assignment1",
      title: "Stanford CS336 Assignment 1: Basics",
      course: "Stanford CS336",
      sourceUrl: "https://github.com/stanford-cs336/assignment1-basics",
      readinessStatus: "runnable",
      runnerTarget: "pyodide",
      rubricKind: "python",
      nextAction: "mount-content",
      summary: "mount required assignment files and datasets before launch",
      launchable: false,
      missingCapabilities: [],
      missingRequiredFiles: ["/assignments/cs336-assignment1/rubric.py"],
      missingDatasets: ["tiny"],
      skippedOptionalPaths: [
        "/assignments/cs336-assignment1/assignment.py",
        "/assignments/cs336-assignment1/reference.py",
      ],
      selectedCapabilities: ["pyodide"],
      simulatedCapabilities: [],
      externalCapabilities: [],
      cacheStrategies: ["invalid-hash"],
      externalRunnerRequired: false,
      messages: [
        "missing required file: /assignments/cs336-assignment1/rubric.py",
        "missing dataset: tiny",
      ],
    });

    expect(
      createAssignmentPlatformHandoff(result.profile, report, {
        files: {
          "/assignments/cs336-assignment1/rubric.py": "print('ok')",
        },
        datasets: { tiny: "fixture" },
      }),
    ).toEqual(
      expect.objectContaining({
        nextAction: "run-pyodide",
        summary: "assignment can launch in Pyodide",
        launchable: true,
        missingRequiredFiles: [],
        missingDatasets: [],
        messages: ["ready for Pyodide rubric runner"],
      }),
    );
  });

  it("creates a verified platform handoff that blocks bad dataset hashes", async () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "python_runtime",
          kind: "capability",
          options: { requires: ["pyodide"] },
        },
      ],
      datasets: [
        {
          name: "tiny",
          url: "/fixtures/tiny.txt",
          hash: "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const report = createAssignmentPreflightReport(
      result.profile,
      createAssignmentCapabilityEnvironment({ browserCapabilities: ["pyodide"] }),
    );
    const files = {
      "/assignments/cs336-assignment1/rubric.py": "print('ok')",
    };

    await expect(
      createVerifiedAssignmentPlatformHandoff(result.profile, report, {
        files,
        datasets: { tiny: "abcd" },
      }),
    ).resolves.toEqual(
      expect.objectContaining({
        nextAction: "verify-content",
        launchable: false,
        hashOk: false,
        messages: ["dataset hash mismatch: tiny"],
      }),
    );

    await expect(
      createVerifiedAssignmentPlatformHandoff(result.profile, report, {
        files,
        datasets: { tiny: "abc" },
      }),
    ).resolves.toEqual(
      expect.objectContaining({
        nextAction: "run-pyodide",
        launchable: true,
        hashOk: true,
        messages: ["ready for Pyodide rubric runner"],
      }),
    );
  });

  it("creates a platform issue draft from a handoff", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "python_runtime",
          kind: "capability",
          options: { requires: ["pyodide"] },
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const report = createAssignmentPreflightReport(
      result.profile,
      createAssignmentCapabilityEnvironment({ browserCapabilities: ["pyodide"] }),
    );
    const handoff = createAssignmentPlatformHandoff(result.profile, report, {
      files: {},
    });

    expect(createAssignmentPlatformIssueDraft(result.profile, handoff)).toEqual({
      title: "BrowserGrad handoff: Stanford CS336 Assignment 1: Basics",
      labels: [
        "browsergrad-handoff",
        "next:mount-content",
        "readiness:runnable",
        "runner:pyodide",
      ],
      body: [
        "## BrowserGrad Handoff",
        "",
        "- Assignment: Stanford CS336 Assignment 1: Basics",
        "- Profile: cs336-assignment1",
        "- Source: https://github.com/stanford-cs336/assignment1-basics",
        "- Readiness: runnable",
        "- Runner: pyodide",
        "- Next action: mount-content",
        "- Launchable: no",
        "",
        "## Messages",
        "",
        "- missing required file: /assignments/cs336-assignment1/rubric.py",
        "- missing dataset: tiny",
        "",
        "## Missing Content",
        "",
        "- Required file: /assignments/cs336-assignment1/rubric.py",
        "- Dataset: tiny",
        "",
        "## Capabilities",
        "",
        "- Selected: pyodide",
        "- Simulated: none",
        "- External: none",
      ].join("\n"),
    });
  });

  it("includes hash checks in issue drafts for verified handoffs", async () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "python_runtime",
          kind: "capability",
          options: { requires: ["pyodide"] },
        },
      ],
      datasets: [
        {
          name: "tiny",
          url: "/fixtures/tiny.txt",
          hash: "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const report = createAssignmentPreflightReport(
      result.profile,
      createAssignmentCapabilityEnvironment({ browserCapabilities: ["pyodide"] }),
    );
    const handoff = await createVerifiedAssignmentPlatformHandoff(
      result.profile,
      report,
      {
        files: { "/assignments/cs336-assignment1/rubric.py": "print('ok')" },
        datasets: { tiny: "abcd" },
      },
    );
    const draft = createAssignmentPlatformIssueDraft(result.profile, handoff);

    expect(draft.labels).toContain("next:verify-content");
    expect(draft.body).toContain("## Hash Checks");
    expect(draft.body).toContain("- tiny: mismatch");
    expect(draft.body).toContain(
      "expected sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
    expect(draft.body).toContain("actual sha256:");
  });

  it("creates a cross-profile capability catalog for platform substrate triage", () => {
    const pythonResult = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "python_runtime",
          kind: "capability",
          options: { requires: ["pyodide", "torch-compat"] },
        },
      ],
    });
    const kernelResult = parseAssignmentProfile({
      ...VALID_PROFILE,
      id: "kernel-lab-smoke",
      gates: [
        {
          name: "kernel_path",
          kind: "capability",
          options: {
            requires: ["structured-assertions"],
            any_of: [["webgpu", "wgsl-kernel"], ["native-cuda-external"]],
          },
        },
      ],
    });
    expect(pythonResult.ok).toBe(true);
    expect(kernelResult.ok).toBe(true);
    if (!pythonResult.ok || !kernelResult.ok) return;

    const catalog = createAssignmentCapabilityCatalog([
      pythonResult.profile,
      kernelResult.profile,
    ]);

    expect(catalog.capabilities).toContainEqual({
      capability: "pyodide",
      profiles: ["cs336-assignment1"],
      requiredBy: [{ profileId: "cs336-assignment1", gate: "python_runtime" }],
      alternativeIn: [],
    });
    expect(catalog.capabilities).toContainEqual({
      capability: "webgpu",
      profiles: ["kernel-lab-smoke"],
      requiredBy: [],
      alternativeIn: [
        {
          profileId: "kernel-lab-smoke",
          gate: "kernel_path",
          group: ["webgpu", "wgsl-kernel"],
        },
      ],
    });
    expect(catalog.capabilities.map((entry) => entry.capability)).toEqual([
      "native-cuda-external",
      "pyodide",
      "structured-assertions",
      "torch-compat",
      "webgpu",
      "wgsl-kernel",
    ]);
  });

  it("verifies benchmark preflight matrix dataset hashes before mounting", async () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "python_runtime",
          kind: "capability",
          options: { requires: ["pyodide"] },
        },
      ],
      datasets: [
        {
          name: "tiny",
          url: "/fixtures/tiny.txt",
          hash: "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const environment = createAssignmentCapabilityEnvironment({
      browserCapabilities: ["pyodide"],
    });
    const files = {
      "/assignments/cs336-assignment1/rubric.py": "print('ok')",
    };

    const wrongMatrix = await createVerifiedAssignmentBenchmarkPreflightMatrix(
      [result.profile],
      environment,
      { files, datasets: { tiny: "abcd" } },
    );

    expect(wrongMatrix.ok).toBe(false);
    expect(wrongMatrix.rows[0]).toEqual(
      expect.objectContaining({
        id: "cs336-assignment1",
        ok: false,
        contentOk: true,
        hashOk: false,
        hashChecks: [
          expect.objectContaining({
            name: "tiny",
            status: "mismatch",
            ok: false,
            expected:
              "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            actual: expect.stringMatching(/^sha256:/),
          }),
        ],
      }),
    );

    const correctMatrix = await createVerifiedAssignmentBenchmarkPreflightMatrix(
      [result.profile],
      environment,
      { files, datasets: { tiny: "abc" } },
    );

    expect(correctMatrix.ok).toBe(true);
    expect(correctMatrix.rows[0]).toEqual(
      expect.objectContaining({
        ok: true,
        contentOk: true,
        hashOk: true,
        hashChecks: [
          expect.objectContaining({
            name: "tiny",
            status: "match",
            ok: true,
          }),
        ],
      }),
    );
  });
});
