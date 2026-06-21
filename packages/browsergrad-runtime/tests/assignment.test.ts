import { describe, expect, it } from "vitest";
import {
  assignmentRubricKind,
  assignmentRunReadiness,
  createAssignmentPreflightReport,
  createAssignmentMountPlan,
  createAssignmentRunPlan,
  createAssignmentRubricExecRequest,
  evaluateAssignmentCapabilities,
  materializeAssignmentMountPlan,
  parseAssignmentProfile,
  profileOracleJsModules,
  requiredAssignmentCapabilities,
  runAssignmentJavascriptRubric,
  runAssignmentRubric,
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
    expect(report.requiredCapabilities).toEqual(["pyodide"]);
    expect(report.mountPlan.datasets).toEqual([
      {
        name: "tiny",
        url: "/fixtures/tiny.txt",
        hash: "sha256:abc",
        mountPath: "/assignments/cs336-assignment1/fixtures/datasets/tiny.txt",
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

  it("materializes provided assignment mount contents into SessionFS", async () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const runPlan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });
    const mountPlan = createAssignmentMountPlan(runPlan);
    const writes: Array<{ path: string; content: string }> = [];

    const materialized = await materializeAssignmentMountPlan(
      {
        async write(path, content) {
          writes.push({ path, content });
        },
        async read(path) {
          return writes.find((write) => write.path === path)?.content ?? "";
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

  it("rejects missing required mount content before writing optional files", async () => {
    const result = parseAssignmentProfile(VALID_PROFILE);
    expect(result.ok).toBe(true);
    if (!result.ok) return;

    const runPlan = createAssignmentRunPlan(result.profile, {
      capabilities: ["pyodide"],
    });
    const mountPlan = createAssignmentMountPlan(runPlan);
    const writes: Array<{ path: string; content: string }> = [];

    await expect(
      materializeAssignmentMountPlan(
        {
          async write(path, content) {
            writes.push({ path, content });
          },
          async read() {
            return "";
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
    const writes: Array<{ path: string; content: string }> = [];

    await expect(
      materializeAssignmentMountPlan(
        {
          async write(path, content) {
            writes.push({ path, content });
          },
          async read() {
            return "";
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
    const writes: Array<{ path: string; content: string }> = [];

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
