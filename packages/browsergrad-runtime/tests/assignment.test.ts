import { describe, expect, it } from "vitest";
import {
  assignmentRubricKind,
  assignmentRunReadiness,
  assignmentRunnerRoute,
  createAssignmentCapabilityEnvironment,
  createAssignmentCapabilityCatalog,
  createAssignmentExternalRunnerRequest,
  createAssignmentBenchmarkPreflightMatrix,
  createVerifiedAssignmentBenchmarkPreflightMatrix,
  createAssignmentPlatformHandoff,
  createVerifiedAssignmentPlatformHandoff,
  createAssignmentPreflightReport,
  createAssignmentMountPreflightReport,
  createAssignmentMountPlan,
  createAssignmentDatasetCachePlan,
  createAssignmentRunPlan,
  createAssignmentRubricExecRequest,
  evaluateAssignmentCapabilities,
  evaluateAssignmentMountContents,
  materializeAssignmentMountPlan,
  parseAssignmentProfile,
  profileOracleJsModules,
  requiredAssignmentCapabilities,
  runAssignmentJavascriptRubric,
  runAssignmentRubric,
  verifyAssignmentMountContentHashes,
} from "../src/index";

const VALID_PROFILE = {
  id: "cs336-assignment1",
  version: "1.0.0",
  requires_browsergrad: "^0.1.0",
  metadata: {
    title: "Stanford CS336 Assignment 1: Basics",
    course: "Stanford CS336",
    source_url: "https://github.com/stanford-cs336/assignment1-basics",
    lecture_urls: ["https://www.youtube.com/watch?v=example"],
    tags: ["language-modeling", "tokenization"],
  },
  runtime_packages: ["numpy", "regex", "pytest"],
  files: {
    root: "/assignments/cs336-assignment1",
    rubric_path: "rubric.py",
    starter_path: "assignment.py",
    reference_path: "reference.py",
    fixtures_path: "fixtures",
  },
  timeouts: {
    setup_ms: 10_000,
    test_ms: 30_000,
    worker_ms: 60_000,
  },
  allowed_tests: ["test_train_bpe_tiny", "test_encode_iterable_streams"],
  oracles: [
    {
      name: "_bg_tokenizers",
      js_module: "/assets/tokenizer-oracle.js",
      export_name: "oracle",
    },
  ],
  gates: [
    {
      name: "browser_runtime",
      kind: "capability",
      options: { requires: ["pyodide"] },
    },
    {
      name: "encode_iterable_streaming",
      kind: "streaming",
      options: { max_chunks_before_first_yield: 2 },
    },
  ],
  datasets: [{ name: "tiny", url: "/fixtures/tiny.txt", hash: "sha256:abc" }],
};

describe("parseAssignmentProfile", () => {
  it("accepts a complete assignment profile", () => {
    const result = parseAssignmentProfile(VALID_PROFILE);

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.profile.id).toBe("cs336-assignment1");
      expect(result.profile.metadata?.course).toBe("Stanford CS336");
      expect(result.profile.metadata?.lecture_urls).toEqual([
        "https://www.youtube.com/watch?v=example",
      ]);
      expect(result.profile.runtime_packages).toEqual(["numpy", "regex", "pytest"]);
      expect(result.profile.gates[0]?.kind).toBe("capability");
      expect(result.profile.gates[1]?.kind).toBe("streaming");
    }
  });

  it("rejects malformed profile shape", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      id: "Bad_ID",
      files: { root: "/x" },
      timeouts: { test_ms: -1 },
      gates: [{ name: "bad", kind: "memory-rss" }],
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors.some((error) => error.includes("id"))).toBe(true);
      expect(result.errors.some((error) => error.includes("rubric_path"))).toBe(true);
      expect(result.errors.some((error) => error.includes("test_ms"))).toBe(true);
      expect(result.errors.some((error) => error.includes("kind"))).toBe(true);
    }
  });

  it("rejects malformed capability gate options", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "bad_capability_options",
          kind: "capability",
          options: {
            requires: ["pyodide", 123],
            any_of: [["webgpu"], "native-cuda"],
            message: false,
          },
        },
      ],
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors).toEqual([
        "gates[0].options.requires[1]: must be a string",
        "gates[0].options.any_of[1]: must be a string array",
        "gates[0].options.message: must be a string when present",
      ]);
    }
  });

  it("rejects malformed browser-safe behavioral gate options", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "bad_streaming",
          kind: "streaming",
          options: {
            max_chunks_before_first_yield: -1,
            chunk_count: "many",
          },
        },
        {
          name: "bad_forbidden_read",
          kind: "forbidden-read",
          options: {
            methods: ["read", 42],
          },
        },
        {
          name: "bad_timeout",
          kind: "timeout",
          options: {
            timeout_ms: 1.5,
          },
        },
      ],
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.errors).toEqual([
        "gates[0].options.max_chunks_before_first_yield: must be a non-negative integer",
        "gates[0].options.chunk_count: must be a non-negative integer when present",
        "gates[1].options.methods[1]: must be a string",
        "gates[2].options.timeout_ms: must be a non-negative integer",
      ]);
    }
  });

  it("maps oracle specs to runtime JS module registrations", () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    expect(profileOracleJsModules(result.profile)).toEqual([
      {
        name: "_bg_tokenizers",
        importURL: "/assets/tokenizer-oracle.js",
        exportName: "oracle",
      },
    ]);
  });

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

  it("evaluates required and alternative capability gates", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "flash_attention_path",
          kind: "capability",
          options: {
            requires: ["pyodide", "torch-compat"],
            any_of: [["webgpu", "wgsl-kernel"], ["triton-compatible"]],
            message: "FlashAttention labs need a browser or native kernel path.",
          },
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    expect(
      evaluateAssignmentCapabilities(result.profile, {
        capabilities: ["pyodide", "torch-compat", "webgpu"],
      }),
    ).toEqual({
      ok: false,
      satisfiedCapabilities: ["pyodide", "torch-compat"],
      missingCapabilities: ["triton-compatible", "wgsl-kernel"],
      capabilityModes: {},
      gates: [
        {
          name: "flash_attention_path",
          ok: false,
          status: "blocked",
          requires: ["pyodide", "torch-compat"],
          anyOf: [["webgpu", "wgsl-kernel"], ["triton-compatible"]],
          selectedAnyOf: [],
          selectedCapabilities: [],
          satisfiedCapabilities: ["pyodide", "torch-compat"],
          missingRequired: [],
          missingAnyOf: [["wgsl-kernel"], ["triton-compatible"]],
          message: "FlashAttention labs need a browser or native kernel path.",
        },
      ],
    });

    expect(
      evaluateAssignmentCapabilities(result.profile, {
        capabilities: ["pyodide", "torch-compat", "webgpu", "wgsl-kernel"],
      }).ok,
    ).toBe(true);
  });

  it("selects the strongest satisfied capability route for each gate", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "gpu_path",
          kind: "capability",
          options: {
            requires: ["webgpu"],
            any_of: [["cuda-compatible-subset"], ["wgsl-kernel"], ["native-cuda-external"]],
          },
        },
        {
          name: "native_only_path",
          kind: "capability",
          options: {
            any_of: [["native-cuda-external"]],
          },
        },
        {
          name: "missing_path",
          kind: "capability",
          options: {
            any_of: [["worker-mesh", "distributed-simulator"]],
          },
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const evaluation = evaluateAssignmentCapabilities(result.profile, {
      capabilities: [
        "cuda-compatible-subset",
        "native-cuda-external",
        "webgpu",
        "wgsl-kernel",
      ],
      capabilityModes: {
        "cuda-compatible-subset": "simulated",
        "native-cuda-external": "external",
        webgpu: "browser",
        "wgsl-kernel": "browser",
      },
    });

    expect(evaluation.gates.map((gate) => ({
      name: gate.name,
      status: gate.status,
      selectedAnyOf: gate.selectedAnyOf,
      selectedCapabilities: gate.selectedCapabilities,
    }))).toEqual([
      {
        name: "gpu_path",
        status: "runnable",
        selectedAnyOf: ["wgsl-kernel"],
        selectedCapabilities: ["webgpu", "wgsl-kernel"],
      },
      {
        name: "native_only_path",
        status: "external-only",
        selectedAnyOf: ["native-cuda-external"],
        selectedCapabilities: ["native-cuda-external"],
      },
      {
        name: "missing_path",
        status: "blocked",
        selectedAnyOf: [],
        selectedCapabilities: [],
      },
    ]);
  });

  it("lists required capabilities deterministically and ignores behavioral gates", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      gates: [
        {
          name: "kernel_path",
          kind: "capability",
          options: {
            requires: ["pyodide", "torch-compat", "pyodide"],
            any_of: [["webgpu", "wgsl-kernel"], ["native-cuda-external"]],
          },
        },
        {
          name: "streaming_behavior",
          kind: "streaming",
          options: { max_chunks_before_first_yield: 2 },
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    expect(requiredAssignmentCapabilities(result.profile)).toEqual([
      "native-cuda-external",
      "pyodide",
      "torch-compat",
      "webgpu",
      "wgsl-kernel",
    ]);
  });

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

  it("creates a deterministic mount plan from a run plan", () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });

    expect(createAssignmentMountPlan(plan)).toEqual({
      root: "/assignments/cs336-assignment1",
      files: [
        {
          role: "rubric",
          path: "/assignments/cs336-assignment1/rubric.py",
          required: true,
        },
        {
          role: "starter",
          path: "/assignments/cs336-assignment1/assignment.py",
          required: false,
        },
        {
          role: "reference",
          path: "/assignments/cs336-assignment1/reference.py",
          required: false,
        },
      ],
      datasets: [
        {
          name: "tiny",
          url: "/fixtures/tiny.txt",
          hash: "sha256:abc",
          mountPath: "/assignments/cs336-assignment1/fixtures/datasets/tiny.txt",
        },
      ],
    });
  });

  it("creates deterministic dataset cache entries from mount plans", () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      datasets: [
        {
          name: "hashed",
          url: "/fixtures/hashed.bin?download=1",
          hash: "sha256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        },
        {
          name: "source-only",
          url: "https://example.test/data/tiny.txt?version=1",
        },
        {
          name: "bad-hash",
          url: "/fixtures/bad.txt",
          hash: "sha256:replace-with-fixture-hash",
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });
    const mountPlan = createAssignmentMountPlan(plan);

    expect(createAssignmentDatasetCachePlan(mountPlan)).toEqual({
      root: "/assignments/cs336-assignment1",
      datasets: [
        {
          name: "hashed",
          url: "/fixtures/hashed.bin?download=1",
          hash: "sha256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
          mountPath: "/assignments/cs336-assignment1/fixtures/datasets/hashed.bin",
          strategy: "content-addressed",
          cacheKey: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
          cachePath: "datasets/sha256/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        },
        {
          name: "source-only",
          url: "https://example.test/data/tiny.txt?version=1",
          mountPath: "/assignments/cs336-assignment1/fixtures/datasets/tiny.txt",
          strategy: "source-addressed",
          cacheKey: "url:https://example.test/data/tiny.txt?version=1",
          cachePath: "datasets/url/https%3A%2F%2Fexample.test%2Fdata%2Ftiny.txt%3Fversion%3D1",
        },
        {
          name: "bad-hash",
          url: "/fixtures/bad.txt",
          hash: "sha256:replace-with-fixture-hash",
          mountPath: "/assignments/cs336-assignment1/fixtures/datasets/bad.txt",
          strategy: "invalid-hash",
          cacheKey: "url:/fixtures/bad.txt",
          cachePath: "datasets/url/%2Ffixtures%2Fbad.txt",
        },
      ],
    });
  });

  it("materializes provided assignment mount contents into SessionFS", async () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const runPlan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });
    const mountPlan = createAssignmentMountPlan(runPlan);
    const writes: Array<{ path: string; content: string | Uint8Array }> = [];

    const materialized = await materializeAssignmentMountPlan(
      {
        async write(path, content) {
          writes.push({ path, content });
        },
        async read(path) {
          const content = writes.find((write) => write.path === path)?.content;
          return typeof content === "string" ? content : "";
        },
        async readBytes(path) {
          const content = writes.find((write) => write.path === path)?.content;
          return typeof content === "string" || content === undefined
            ? new Uint8Array()
            : content;
        },
      },
      mountPlan,
      {
        files: {
          "/assignments/cs336-assignment1/rubric.py": "print('rubric')",
          "/assignments/cs336-assignment1/assignment.py": "answer = 42",
        },
        datasets: {
          tiny: "fixture text",
        },
      },
    );

    expect(materialized).toEqual({
      writtenPaths: [
        "/assignments/cs336-assignment1/rubric.py",
        "/assignments/cs336-assignment1/assignment.py",
        "/assignments/cs336-assignment1/fixtures/datasets/tiny.txt",
      ],
      skippedOptionalPaths: ["/assignments/cs336-assignment1/reference.py"],
    });
    expect(writes).toEqual([
      {
        path: "/assignments/cs336-assignment1/rubric.py",
        content: "print('rubric')",
      },
      {
        path: "/assignments/cs336-assignment1/assignment.py",
        content: "answer = 42",
      },
      {
        path: "/assignments/cs336-assignment1/fixtures/datasets/tiny.txt",
        content: "fixture text",
      },
    ]);
  });

  it("evaluates assignment mount contents without writing to SessionFS", () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const runPlan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });
    const mountPlan = createAssignmentMountPlan(runPlan);

    expect(
      evaluateAssignmentMountContents(mountPlan, {
        files: {
          "/assignments/cs336-assignment1/assignment.py": "answer = 42",
        },
      }),
    ).toEqual({
      ok: false,
      writablePaths: ["/assignments/cs336-assignment1/assignment.py"],
      missingRequiredFiles: ["/assignments/cs336-assignment1/rubric.py"],
      missingDatasets: ["tiny"],
      skippedOptionalPaths: ["/assignments/cs336-assignment1/reference.py"],
    });

    expect(
      evaluateAssignmentMountContents(mountPlan, {
        files: {
          "/assignments/cs336-assignment1/rubric.py": "print('rubric')",
        },
        datasets: {
          tiny: "fixture text",
        },
      }).ok,
    ).toBe(true);
  });

  it("verifies mounted dataset hashes for text and binary contents", async () => {
    const textSha256 =
      "5cb72f90e968922d30557d0af8f719d21f61792becaa87eb32477767d739dc0b";
    const binarySha256 =
      "26a66b061e8f48f39927c312f25293959729eee95978e2892d49d3512a5cc092";
    const wrongSha256 =
      "a40d856f7bd138b40fc924da7f59edc8edb9e4a749994ca49a4a5e5f7a32602d";
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      datasets: [
        {
          name: "tiny",
          url: "/fixtures/tiny.txt",
          hash: `sha256:${textSha256}`,
        },
        {
          name: "tiny-bin",
          url: "/fixtures/tiny.bin",
          hash: `sha256:${binarySha256}`,
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const mountPlan = createAssignmentMountPlan(
      createAssignmentRunPlan(result.profile, { capabilities: ["pyodide"] }),
    );

    await expect(
      verifyAssignmentMountContentHashes(mountPlan, {
        files: { "/assignments/cs336-assignment1/rubric.py": "print('rubric')" },
        datasets: {
          tiny: "fixture text",
          "tiny-bin": Uint8Array.of(0, 1, 255),
        },
      }),
    ).resolves.toEqual({
      ok: true,
      checks: [
        {
          name: "tiny",
          path: "/assignments/cs336-assignment1/fixtures/datasets/tiny.txt",
          algorithm: "sha256",
          expected: `sha256:${textSha256}`,
          actual: `sha256:${textSha256}`,
          ok: true,
          status: "match",
        },
        {
          name: "tiny-bin",
          path: "/assignments/cs336-assignment1/fixtures/datasets/tiny.bin",
          algorithm: "sha256",
          expected: `sha256:${binarySha256}`,
          actual: `sha256:${binarySha256}`,
          ok: true,
          status: "match",
        },
      ],
    });

    await expect(
      verifyAssignmentMountContentHashes(mountPlan, {
        files: { "/assignments/cs336-assignment1/rubric.py": "print('rubric')" },
        datasets: {
          tiny: "wrong fixture",
          "tiny-bin": Uint8Array.of(0, 1, 255),
        },
      }),
    ).resolves.toMatchObject({
      ok: false,
      checks: [
        {
          name: "tiny",
          expected: `sha256:${textSha256}`,
          actual: `sha256:${wrongSha256}`,
          ok: false,
          status: "mismatch",
        },
        {
          name: "tiny-bin",
          ok: true,
          status: "match",
        },
      ],
    });
  });

  it("bundles mount content readiness and hash verification", async () => {
    const textSha256 =
      "5cb72f90e968922d30557d0af8f719d21f61792becaa87eb32477767d739dc0b";
    const wrongSha256 =
      "a40d856f7bd138b40fc924da7f59edc8edb9e4a749994ca49a4a5e5f7a32602d";
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      datasets: [
        {
          name: "tiny",
          url: "/fixtures/tiny.txt",
          hash: `sha256:${textSha256}`,
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const mountPlan = createAssignmentMountPlan(
      createAssignmentRunPlan(result.profile, { capabilities: ["pyodide"] }),
    );

    await expect(
      createAssignmentMountPreflightReport(mountPlan, {
        files: { "/assignments/cs336-assignment1/rubric.py": "print('rubric')" },
        datasets: { tiny: "wrong fixture" },
      }),
    ).resolves.toMatchObject({
      ok: false,
      content: {
        ok: true,
        missingRequiredFiles: [],
        missingDatasets: [],
        skippedOptionalPaths: ["/assignments/cs336-assignment1/assignment.py",
          "/assignments/cs336-assignment1/reference.py"],
      },
      hashes: {
        ok: false,
        checks: [
          {
            name: "tiny",
            expected: `sha256:${textSha256}`,
            actual: `sha256:${wrongSha256}`,
            ok: false,
            status: "mismatch",
          },
        ],
      },
    });
  });

  it("materializes binary assignment dataset contents", async () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const runPlan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });
    const mountPlan = createAssignmentMountPlan(runPlan);
    const writes: Array<{ path: string; content: string | Uint8Array }> = [];
    const binaryFixture = Uint8Array.of(0x50, 0x54, 0x00, 0xff);

    const materialized = await materializeAssignmentMountPlan(
      {
        async write(path, content) {
          writes.push({ path, content });
        },
        async read() {
          return "";
        },
        async readBytes() {
          return new Uint8Array();
        },
      },
      mountPlan,
      {
        files: {
          "/assignments/cs336-assignment1/rubric.py": "print('rubric')",
        },
        datasets: {
          tiny: binaryFixture,
        },
      },
    );

    expect(materialized.writtenPaths).toEqual([
      "/assignments/cs336-assignment1/rubric.py",
      "/assignments/cs336-assignment1/fixtures/datasets/tiny.txt",
    ]);
    expect(writes.at(-1)).toEqual({
      path: "/assignments/cs336-assignment1/fixtures/datasets/tiny.txt",
      content: binaryFixture,
    });
  });

  it("rejects missing required mount content before writing optional files", async () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const runPlan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });
    const mountPlan = createAssignmentMountPlan(runPlan);
    const writes: Array<{ path: string; content: string | Uint8Array }> = [];

    await expect(
      materializeAssignmentMountPlan(
        {
          async write(path, content) {
            writes.push({ path, content });
          },
          async read() {
            return "";
          },
          async readBytes() {
            return new Uint8Array();
          },
        },
        mountPlan,
        {
          files: {
            "/assignments/cs336-assignment1/assignment.py": "answer = 42",
          },
          datasets: {
            tiny: "fixture text",
          },
        },
      ),
    ).rejects.toThrow(
      "missing required assignment file content: /assignments/cs336-assignment1/rubric.py",
    );
    expect(writes).toEqual([]);
  });

  it("rejects missing dataset mount content before writing files", async () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const runPlan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });
    const mountPlan = createAssignmentMountPlan(runPlan);
    const writes: Array<{ path: string; content: string | Uint8Array }> = [];

    await expect(
      materializeAssignmentMountPlan(
        {
          async write(path, content) {
            writes.push({ path, content });
          },
          async read() {
            return "";
          },
          async readBytes() {
            return new Uint8Array();
          },
        },
        mountPlan,
        {
          files: {
            "/assignments/cs336-assignment1/rubric.py": "print('rubric')",
          },
        },
      ),
    ).rejects.toThrow("missing assignment dataset content: tiny");
    expect(writes).toEqual([]);
  });

  it("rejects assignment rubric runs with failed preflight before mounting files", async () => {
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
    const writes: Array<{ path: string; content: string | Uint8Array }> = [];

    await expect(
      runAssignmentRubric(
        {
          fs: {
            async write(path, content) {
              writes.push({ path, content });
            },
            async read() {
              return "";
            },
            async readBytes() {
              return new Uint8Array();
            },
          },
          async exec() {
            throw new Error("exec should not run");
          },
        },
        plan,
        {
          files: {
            "/assignments/cs336-assignment1/rubric.py": "print('rubric')",
          },
          datasets: {
            tiny: "fixture text",
          },
        },
      ),
    ).rejects.toThrow(
      "cannot create rubric exec request; missing assignment capabilities: webgpu",
    );
    expect(writes).toEqual([]);
  });

  it("runs a JavaScript rubric with mounted text, assignment context, oracles, and assertions", async () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      id: "gpu-puzzles",
      runtime_packages: [],
      files: {
        root: "/assignments/gpu-puzzles",
        rubric_path: "rubric.js",
        starter_path: "student.js",
        fixtures_path: "fixtures",
      },
      allowed_tests: ["test_js_rubric"],
      oracles: [
        {
          name: "_bg_gpu_oracle",
          js_module: "/assignments/gpu-puzzles/oracles/gpu-puzzles.js",
          export_name: "oracle",
        },
      ],
      gates: [
        {
          name: "js_rubric_runtime",
          kind: "capability",
          options: { requires: ["javascript-rubric"] },
        },
        {
          name: "streaming_contract",
          kind: "streaming",
          options: { max_chunks_before_first_yield: 2 },
        },
      ],
      datasets: [{ name: "tiny", url: "/fixtures/tiny.txt" }],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["javascript-rubric"],
    });

    const run = await runAssignmentJavascriptRubric(
      plan,
      {
        files: {
          "/assignments/gpu-puzzles/rubric.js": "export async function run() {}",
          "/assignments/gpu-puzzles/student.js": "export const answer = 42;",
        },
        datasets: {
          tiny: "fixture text",
        },
      },
      async (ctx) => {
        const oracle = ctx.oracle<{ score(input: string): number }>("_bg_gpu_oracle");
        const fixture = ctx.readText("/assignments/gpu-puzzles/fixtures/datasets/tiny.txt");
        ctx.emitJson("js_rubric_context", {
          id: ctx.id,
          root: ctx.root,
          fixturesPath: ctx.fixturesPath,
          allowedTests: ctx.allowedTests,
          behavioralGates: ctx.behavioralGates,
          rubricSource: ctx.readText("/assignments/gpu-puzzles/rubric.js"),
          studentSource: ctx.readText("/assignments/gpu-puzzles/student.js"),
        });
        if (fixture === "fixture text" && oracle.score(fixture) === 24) {
          ctx.assertPass("test_js_rubric");
        } else {
          ctx.assertFail("test_js_rubric", "JS rubric mismatch", {
            expected: "fixture text / 24",
            actual: `${fixture} / ${oracle.score(fixture)}`,
          });
        }
      },
      {
        oracles: {
          _bg_gpu_oracle: {
            score: (input: string) => input.length * 2,
          },
        },
      },
    );

    expect(run.mount).toEqual({
      writtenPaths: [
        "/assignments/gpu-puzzles/rubric.js",
        "/assignments/gpu-puzzles/student.js",
        "/assignments/gpu-puzzles/fixtures/datasets/tiny.txt",
      ],
      skippedOptionalPaths: [],
    });
    expect(run.assertions).toEqual([
      { kind: "pass", name: "test_js_rubric" },
    ]);
    expect(run.artifacts).toEqual([
      {
        kind: "json",
        name: "js_rubric_context",
        data: {
          id: "gpu-puzzles",
          root: "/assignments/gpu-puzzles",
          fixturesPath: "/assignments/gpu-puzzles/fixtures",
          allowedTests: ["test_js_rubric"],
          behavioralGates: [
            {
              name: "streaming_contract",
              kind: "streaming",
              options: { max_chunks_before_first_yield: 2 },
            },
          ],
          rubricSource: "export async function run() {}",
          studentSource: "export const answer = 42;",
        },
      },
    ]);
  });

  it("rejects JavaScript rubric runs with failed preflight before invoking the rubric", async () => {
    const result = parseAssignmentProfile({
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
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["javascript-rubric"],
    });
    let invoked = false;

    await expect(
      runAssignmentJavascriptRubric(
        plan,
        {
          files: {
            "/assignments/cs336-assignment1/rubric.js": "export async function run() {}",
          },
          datasets: {
            tiny: "fixture text",
          },
        },
        () => {
          invoked = true;
        },
      ),
    ).rejects.toThrow(
      "cannot run JavaScript rubric; missing assignment capabilities: webgpu",
    );
    expect(invoked).toBe(false);
  });

  it("passes browser substrate objects to JavaScript rubrics", async () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      id: "gpu-puzzles",
      runtime_packages: [],
      files: {
        root: "/assignments/gpu-puzzles",
        rubric_path: "rubric.js",
      },
      allowed_tests: ["test_gpu_substrate"],
      gates: [
        {
          name: "webgpu_path",
          kind: "capability",
          options: { requires: ["javascript-rubric", "webgpu"] },
        },
      ],
      datasets: [],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["javascript-rubric", "webgpu"],
    });

    const run = await runAssignmentJavascriptRubric(
      plan,
      {
        files: {
          "/assignments/gpu-puzzles/rubric.js": "export async function run() {}",
        },
      },
      async (ctx) => {
        const webgpu = ctx.substrate<{ adapter: string }>("webgpu");
        if (webgpu.adapter === "mock-adapter") {
          ctx.assertPass("test_gpu_substrate");
        } else {
          ctx.assertFail("test_gpu_substrate", "wrong substrate");
        }
      },
      {
        substrates: {
          webgpu: { adapter: "mock-adapter" },
        },
      },
    );

    expect(run.assertions).toEqual([
      { kind: "pass", name: "test_gpu_substrate" },
    ]);
  });

  it("lets JavaScript rubrics read binary mounted content", async () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      id: "alignment-snapshots",
      runtime_packages: [],
      files: {
        root: "/assignments/alignment-snapshots",
        rubric_path: "rubric.js",
        fixtures_path: "fixtures",
      },
      allowed_tests: ["test_binary_snapshot"],
      gates: [
        {
          name: "js_snapshot_rubric",
          kind: "capability",
          options: { requires: ["javascript-rubric", "snapshot-oracle"] },
        },
      ],
      datasets: [{ name: "tiny-npz", url: "/fixtures/tiny.npz" }],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["javascript-rubric", "snapshot-oracle"],
    });
    const snapshotBytes = Uint8Array.of(0x50, 0x4b, 0x03, 0x04);

    const run = await runAssignmentJavascriptRubric(
      plan,
      {
        files: {
          "/assignments/alignment-snapshots/rubric.js": "export async function run() {}",
        },
        datasets: {
          "tiny-npz": snapshotBytes,
        },
      },
      (ctx) => {
        const bytes = ctx.readBytes(
          "/assignments/alignment-snapshots/fixtures/datasets/tiny.npz",
        );
        if (bytes[0] === 0x50 && bytes[1] === 0x4b) {
          ctx.assertPass("test_binary_snapshot");
        } else {
          ctx.assertFail("test_binary_snapshot", "wrong bytes");
        }
      },
    );

    expect(run.assertions).toEqual([
      { kind: "pass", name: "test_binary_snapshot" },
    ]);
  });

  it("throws a clear error when a JavaScript rubric requires a missing substrate", async () => {
    const result = parseAssignmentProfile({
      ...VALID_PROFILE,
      files: {
        root: "/assignments/gpu-puzzles",
        rubric_path: "rubric.js",
      },
      gates: [
        {
          name: "webgpu_path",
          kind: "capability",
          options: { requires: ["javascript-rubric", "webgpu"] },
        },
      ],
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const plan = createAssignmentRunPlan(result.profile, {
      capabilities: ["javascript-rubric", "webgpu"],
    });

    await expect(
      runAssignmentJavascriptRubric(
        plan,
        {
          files: {
            "/assignments/gpu-puzzles/rubric.js": "export async function run() {}",
          },
          datasets: {
            tiny: "fixture text",
          },
        },
        (ctx) => {
          ctx.substrate("webgpu");
        },
      ),
    ).rejects.toThrow("assignment JavaScript substrate is not registered: webgpu");
  });
});

describe("createAssignmentCapabilityEnvironment", () => {
  it("normalizes browser, simulated, and external capability groups", () => {
    expect(
      createAssignmentCapabilityEnvironment({
        browserCapabilities: ["pyodide", "webgpu", "pyodide"],
        simulatedCapabilities: ["worker-mesh", "distributed-simulator"],
        externalCapabilities: ["native-cuda-external", "webgpu", "worker-mesh"],
      }),
    ).toEqual({
      capabilities: [
        "distributed-simulator",
        "native-cuda-external",
        "pyodide",
        "webgpu",
        "worker-mesh",
      ],
      capabilityModes: {
        "distributed-simulator": "simulated",
        "native-cuda-external": "external",
        pyodide: "browser",
        webgpu: "browser",
        "worker-mesh": "simulated",
      },
    });
  });
});
