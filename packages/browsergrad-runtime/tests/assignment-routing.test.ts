import { describe, expect, it } from "vitest";
import { VALID_PROFILE } from "./assignment-fixtures";
import {
  assignmentRubricKind,
  assignmentRunReadiness,
  assignmentRunnerRoute,
  createAssignmentCapabilityEnvironment,
  createAssignmentExternalRunnerRequest,
  createAssignmentPreflightReport,
  createAssignmentRunPlan,
  createAssignmentRubricExecRequest,
  parseAssignmentProfile,
} from "../src/index";

describe("assignment routing and exec requests", () => {
  it("creates a substrate-neutral run plan from profile data", () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    expect(
      createAssignmentRunPlan(result.profile, {
        capabilities: ["pyodide"],
      }),
    ).toEqual({
      id: "cs336-assignment1",
      profileVersion: "1.0.0",
      requiresBrowsergrad: "^0.1.0",
      ok: true,
      session: {
        packages: ["numpy", "regex", "pytest"],
        jsModules: [
          {
            name: "_bg_tokenizers",
            importURL: "/assets/tokenizer-oracle.js",
            exportName: "oracle",
          },
        ],
      },
      files: {
        root: "/assignments/cs336-assignment1",
        rubricPath: "/assignments/cs336-assignment1/rubric.py",
        starterPath: "/assignments/cs336-assignment1/assignment.py",
        referencePath: "/assignments/cs336-assignment1/reference.py",
        fixturesPath: "/assignments/cs336-assignment1/fixtures",
      },
      execution: {
        allowedTests: ["test_train_bpe_tiny", "test_encode_iterable_streams"],
        setupTimeoutMs: 10_000,
        testTimeoutMs: 30_000,
        workerTimeoutMs: 60_000,
      },
      datasets: [{ name: "tiny", url: "/fixtures/tiny.txt", hash: "sha256:abc" }],
      capabilityEvaluation: {
        ok: true,
        satisfiedCapabilities: ["pyodide"],
        missingCapabilities: [],
        capabilityModes: {},
        gates: [
          {
            name: "browser_runtime",
            ok: true,
            status: "runnable",
            requires: ["pyodide"],
            anyOf: [],
            selectedAnyOf: [],
            selectedCapabilities: ["pyodide"],
            satisfiedCapabilities: ["pyodide"],
            missingRequired: [],
            missingAnyOf: [],
          },
        ],
      },
      behavioralGates: [
        {
          name: "encode_iterable_streaming",
          kind: "streaming",
          options: { max_chunks_before_first_yield: 2 },
        },
      ],
    });
  });

  it("summarizes assignment run readiness from platform capability modes", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "browser_runtime",
          kind: "capability",
          options: { requires: ["pyodide"] },
        },
        {
          name: "distributed_path",
          kind: "capability",
          options: {
            any_of: [
              ["worker-mesh", "distributed-simulator"],
              ["native-distributed-external"],
            ],
          },
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    expect(
      assignmentRunReadiness(
        createAssignmentRunPlan(result.profile, {
          capabilities: ["pyodide", "worker-mesh", "distributed-simulator"],
          capabilityModes: {
            "worker-mesh": "simulated",
            "distributed-simulator": "simulated",
          },
        }),
      ),
    ).toEqual({
      status: "simulated",
      summary: "assignment can run through simulated platform capabilities",
      missingCapabilities: [],
      selectedCapabilities: ["distributed-simulator", "pyodide", "worker-mesh"],
      simulatedCapabilities: ["distributed-simulator", "worker-mesh"],
      externalCapabilities: [],
    });

    expect(
      assignmentRunReadiness(
        createAssignmentRunPlan(result.profile, {
          capabilities: ["pyodide", "native-distributed-external"],
          capabilityModes: {
            "native-distributed-external": "external",
          },
        }),
      ).status,
    ).toBe("external-only");

    expect(
      assignmentRunReadiness(
        createAssignmentRunPlan(result.profile, {
          capabilities: ["pyodide"],
        }),
      ).status,
    ).toBe("blocked");
  });

  it("creates a platform preflight report without executing or mounting code", () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const report = createAssignmentPreflightReport(result.profile, {
      capabilities: ["pyodide"],
      capabilityModes: { pyodide: "browser" },
    });

    expect(report.rubricKind).toBe("python");
    expect(report.readiness).toEqual({
      status: "runnable",
      summary: "assignment can run in the current platform",
      missingCapabilities: [],
      selectedCapabilities: ["pyodide"],
      simulatedCapabilities: [],
      externalCapabilities: [],
    });
    expect(report.runnerRoute).toEqual({
      target: "pyodide",
      readinessStatus: "runnable",
      rubricKind: "python",
      message: "assignment routes to Pyodide rubric runner",
      missingCapabilities: [],
      selectedCapabilities: ["pyodide"],
    });
    expect(report.requiredCapabilities).toEqual(["pyodide"]);
    expect(report.mountPlan.datasets).toEqual([
      {
        name: "tiny",
        url: "/fixtures/tiny.txt",
        hash: "sha256:abc",
        mountPath: "/assignments/cs336-assignment1/fixtures/datasets/tiny.txt",
      },
    ]);
    expect(report.datasetCachePlan.datasets).toEqual([
      {
        name: "tiny",
        url: "/fixtures/tiny.txt",
        hash: "sha256:abc",
        mountPath: "/assignments/cs336-assignment1/fixtures/datasets/tiny.txt",
        strategy: "invalid-hash",
        cacheKey: "url:/fixtures/tiny.txt",
        cachePath: "datasets/url/%2Ffixtures%2Ftiny.txt",
      },
    ]);
    expect(report.plan.ok).toBe(true);
  });

  it("classifies rubric substrate from the run plan path", () => {
    const pythonResult = parseAssignmentProfile(VALID_PROFILE);
    expect(pythonResult.ok).toBe(true);
    if (!pythonResult.ok) return;

    const jsResult = parseAssignmentProfile({
      ...VALID_PROFILE,
      files: {
        ...VALID_PROFILE.files,
        rubric_path: "rubric.js",
      },
    });
    expect(jsResult.ok).toBe(true);
    if (!jsResult.ok) return;

    const unknownResult = parseAssignmentProfile({
      ...VALID_PROFILE,
      files: {
        ...VALID_PROFILE.files,
        rubric_path: "rubric.wasm",
      },
    });
    expect(unknownResult.ok).toBe(true);
    if (!unknownResult.ok) return;

    expect(
      assignmentRubricKind(
        createAssignmentRunPlan(pythonResult.profile, { capabilities: ["pyodide"] }),
      ),
    ).toBe("python");
    expect(
      assignmentRubricKind(
        createAssignmentRunPlan(jsResult.profile, { capabilities: ["pyodide"] }),
      ),
    ).toBe("javascript");
    expect(
      assignmentRubricKind(
        createAssignmentRunPlan(unknownResult.profile, { capabilities: ["pyodide"] }),
      ),
    ).toBe("unknown");
  });

  it("routes run plans to built-in or external runner targets", () => {
    const pythonResult = parseAssignmentProfile(VALID_PROFILE);
    expect(pythonResult.ok).toBe(true);
    if (!pythonResult.ok) return;

    const jsResult = parseAssignmentProfile({
      ...VALID_PROFILE,
      files: {
        ...VALID_PROFILE.files,
        rubric_path: "rubric.js",
      },
      gates: [
        {
          name: "js_rubric_runtime",
          kind: "capability",
          options: { requires: ["javascript-rubric", "webgpu"] },
        },
      ],
    });
    expect(jsResult.ok).toBe(true);
    if (!jsResult.ok) return;

    const unknownResult = parseAssignmentProfile({
      ...VALID_PROFILE,
      files: {
        ...VALID_PROFILE.files,
        rubric_path: "rubric.wasm",
      },
    });
    expect(unknownResult.ok).toBe(true);
    if (!unknownResult.ok) return;

    expect(
      assignmentRunnerRoute(
        createAssignmentRunPlan(pythonResult.profile, { capabilities: ["pyodide"] }),
      ),
    ).toEqual({
      target: "pyodide",
      readinessStatus: "runnable",
      rubricKind: "python",
      message: "assignment routes to Pyodide rubric runner",
      missingCapabilities: [],
      selectedCapabilities: ["pyodide"],
    });

    expect(
      assignmentRunnerRoute(
        createAssignmentRunPlan(jsResult.profile, {
          capabilities: ["javascript-rubric", "webgpu"],
          capabilityModes: {
            "javascript-rubric": "browser",
            webgpu: "browser",
          },
        }),
      ).target,
    ).toBe("javascript");

    expect(
      assignmentRunnerRoute(
        createAssignmentRunPlan(jsResult.profile, {
          capabilities: ["javascript-rubric", "webgpu"],
          capabilityModes: {
            "javascript-rubric": "browser",
            webgpu: "external",
          },
        }),
      ).target,
    ).toBe("external");

    expect(
      assignmentRunnerRoute(
        createAssignmentRunPlan(jsResult.profile, {
          capabilities: ["javascript-rubric"],
        }),
      ).target,
    ).toBe("blocked");

    expect(
      assignmentRunnerRoute(
        createAssignmentRunPlan(unknownResult.profile, { capabilities: ["pyodide"] }),
      ).target,
    ).toBe("unsupported");
  });

  it("creates an external runner request for native-only plans", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      id: "cs149gpt",
      files: {
        root: "/assignments/cs149gpt",
        rubric_path: "rubric.py",
        starter_path: "starter",
        fixtures_path: "fixtures",
      },
      allowed_tests: ["attention_performance_smoke"],
      gates: [
        {
          name: "native_attention_path",
          kind: "capability",
          options: {
            requires: ["attention-oracle"],
            any_of: [["native-cpp-external"], ["browser-cpp-simulator"]],
          },
        },
        {
          name: "student_timeout",
          kind: "timeout",
          options: { timeout_ms: 45_000 },
        },
      ],
      datasets: [
        {
          name: "tiny-attention",
          url: "/fixtures/tiny-attention.npz",
          hash: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(
      result.profile,
      createAssignmentCapabilityEnvironment({
        browserCapabilities: ["attention-oracle"],
        externalCapabilities: ["native-cpp-external"],
      }),
    );

    expect(createAssignmentExternalRunnerRequest(plan)).toEqual({
      id: "cs149gpt",
      profileVersion: "1.0.0",
      requiresBrowsergrad: "^0.1.0",
      route: {
        target: "external",
        readinessStatus: "external-only",
        rubricKind: "python",
        message: "assignment requires external runner capabilities",
        missingCapabilities: [],
        selectedCapabilities: ["attention-oracle", "native-cpp-external"],
      },
      selectedCapabilities: ["attention-oracle", "native-cpp-external"],
      externalCapabilities: ["native-cpp-external"],
      simulatedCapabilities: [],
      environment: {
        BROWSERGRAD_ALLOWED_TESTS_JSON: "[\"attention_performance_smoke\"]",
        BROWSERGRAD_ASSIGNMENT_ID: "cs149gpt",
        BROWSERGRAD_ASSIGNMENT_ROOT: "/assignments/cs149gpt",
        BROWSERGRAD_BEHAVIORAL_GATES_JSON:
          "[{\"name\":\"student_timeout\",\"kind\":\"timeout\",\"options\":{\"timeout_ms\":45000}}]",
        BROWSERGRAD_EXTERNAL_CAPABILITIES_JSON: "[\"native-cpp-external\"]",
        BROWSERGRAD_FIXTURES_PATH: "/assignments/cs149gpt/fixtures",
        BROWSERGRAD_RUNNER_READINESS: "external-only",
        BROWSERGRAD_RUNNER_TARGET: "external",
        BROWSERGRAD_SELECTED_CAPABILITIES_JSON:
          "[\"attention-oracle\",\"native-cpp-external\"]",
        BROWSERGRAD_SIMULATED_CAPABILITIES_JSON: "[]",
      },
      files: {
        root: "/assignments/cs149gpt",
        rubricPath: "/assignments/cs149gpt/rubric.py",
        starterPath: "/assignments/cs149gpt/starter",
        fixturesPath: "/assignments/cs149gpt/fixtures",
      },
      execution: {
        allowedTests: ["attention_performance_smoke"],
        setupTimeoutMs: 10_000,
        testTimeoutMs: 30_000,
        workerTimeoutMs: 60_000,
      },
      mountPlan: {
        root: "/assignments/cs149gpt",
        files: [
          {
            role: "rubric",
            path: "/assignments/cs149gpt/rubric.py",
            required: true,
          },
          {
            role: "starter",
            path: "/assignments/cs149gpt/starter",
            required: false,
          },
        ],
        datasets: [
          {
            name: "tiny-attention",
            url: "/fixtures/tiny-attention.npz",
            hash: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            mountPath: "/assignments/cs149gpt/fixtures/datasets/tiny-attention.npz",
          },
        ],
      },
      datasetCachePlan: {
        root: "/assignments/cs149gpt",
        datasets: [
          {
            name: "tiny-attention",
            url: "/fixtures/tiny-attention.npz",
            hash: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            mountPath: "/assignments/cs149gpt/fixtures/datasets/tiny-attention.npz",
            strategy: "content-addressed",
            cacheKey: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            cachePath: "datasets/sha256/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
          },
        ],
      },
      behavioralGates: [
        {
          name: "student_timeout",
          kind: "timeout",
          options: { timeout_ms: 45_000 },
        },
      ],
    });
  });

  it("includes an external runner request in external preflight reports", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "native_cuda_path",
          kind: "capability",
          options: { requires: ["native-cuda-external"] },
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const report = createAssignmentPreflightReport(
      result.profile,
      createAssignmentCapabilityEnvironment({
        externalCapabilities: ["native-cuda-external"],
      }),
    );

    expect(report.runnerRoute.target).toBe("external");
    expect(report.externalRunnerRequest?.externalCapabilities).toEqual([
      "native-cuda-external",
    ]);
    expect(report.externalRunnerRequest?.mountPlan).toEqual(report.mountPlan);
    expect(report.externalRunnerRequest?.datasetCachePlan).toEqual(
      report.datasetCachePlan,
    );
  });

  it("creates a rubric exec request from a run plan", () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });

    expect(createAssignmentRubricExecRequest(plan)).toEqual({
      code: [
        "import json, os, runpy, sys",
        'assignment_root = "/assignments/cs336-assignment1"',
        "if assignment_root not in sys.path:",
        "    sys.path.insert(0, assignment_root)",
        'os.environ["BROWSERGRAD_ASSIGNMENT_ID"] = "cs336-assignment1"',
        'os.environ["BROWSERGRAD_ASSIGNMENT_ROOT"] = "/assignments/cs336-assignment1"',
        'os.environ["BROWSERGRAD_FIXTURES_PATH"] = "/assignments/cs336-assignment1/fixtures"',
        'os.environ["BROWSERGRAD_ALLOWED_TESTS_JSON"] = "[\\"test_train_bpe_tiny\\",\\"test_encode_iterable_streams\\"]"',
        'os.environ["BROWSERGRAD_BEHAVIORAL_GATES_JSON"] = "[{\\"name\\":\\"encode_iterable_streaming\\",\\"kind\\":\\"streaming\\",\\"options\\":{\\"max_chunks_before_first_yield\\":2}}]"',
        'runpy.run_path("/assignments/cs336-assignment1/rubric.py", run_name="__main__")',
      ].join("\n"),
      timeoutMs: 30_000,
    });
  });

  it("uses the shortest declared rubric watchdog timeout", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      timeouts: {
        setup_ms: 10_000,
        test_ms: 90_000,
        worker_ms: 45_000,
      },
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });

    expect(createAssignmentRubricExecRequest(plan).timeoutMs).toBe(45_000);
  });

  it("uses worker watchdog timeout when no test timeout is declared", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      timeouts: {
        worker_ms: 45_000,
      },
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });

    expect(createAssignmentRubricExecRequest(plan).timeoutMs).toBe(45_000);
  });

  it("rejects Pyodide exec requests for non-Python rubrics", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      files: {
        ...VALID_PROFILE.files,
        rubric_path: "rubric.js",
      },
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });

    expect(() => createAssignmentRubricExecRequest(plan)).toThrow(
      "createAssignmentRubricExecRequest requires a Python rubric path",
    );
  });

  it("rejects rubric exec requests when capability preflight failed", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "kernel_path",
          kind: "capability",
          options: { requires: ["pyodide", "webgpu"] },
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });

    expect(() => createAssignmentRubricExecRequest(plan)).toThrow(
      "cannot create rubric exec request; missing assignment capabilities: webgpu",
    );
  });
});
