import { describe, expect, it } from "vitest";
import {
  createAssignmentMountPlan,
  createAssignmentRunPlan,
  createAssignmentRubricExecRequest,
  evaluateAssignmentCapabilities,
  materializeAssignmentMountPlan,
  parseAssignmentProfile,
  profileOracleJsModules,
  requiredAssignmentCapabilities,
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
      missingCapabilities: ["triton-compatible", "wgsl-kernel"],
      gates: [
        {
          name: "flash_attention_path",
          ok: false,
          requires: ["pyodide", "torch-compat"],
          anyOf: [["webgpu", "wgsl-kernel"], ["triton-compatible"]],
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
        missingCapabilities: [],
        gates: [
          {
            name: "browser_runtime",
            ok: true,
            requires: ["pyodide"],
            anyOf: [],
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
        'os.environ["BROWSERGRAD_ALLOWED_TESTS_JSON"] = "[\\"test_train_bpe_tiny\\",\\"test_encode_iterable_streams\\"]"',
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
});
